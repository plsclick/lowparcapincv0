"""Async terminal application loop and streamed chat presentation."""

import asyncio
import logging
from pathlib import Path
import shlex
from typing import Any, Callable

from lowparllm.ai.coding.agent import CodingAgent
from lowparllm.ai.coding.workspace import CodingWorkspace
from lowparllm.ai.llm.chat_service import ChatService
from lowparllm.ai.llm.ollama_client import OllamaClient, OllamaError
from lowparllm.ai.llm.models import GenerationMetrics
from lowparllm.ai.llm.model_manager import ModelManager
from lowparllm.ai.rag.rag_service import RagService
from lowparllm.ai.rag.retriever import RetrievalError
from lowparllm.ai.rag.vector_store import VectorStoreError
from lowparllm.ai.research.research_service import ResearchBundle, ResearchService
from lowparllm.ai.search.provider import SearchProviderError
from lowparllm.cli.commands import CommandHandler, split_command
from lowparllm.cli.input_handler import InputHandler
from lowparllm.cli.renderer import Renderer
from lowparllm.core.config import AppConfig
from lowparllm.services.system_info import gather_environment

logger = logging.getLogger(__name__)


def _media_path_from_request(request: str, suffixes: set[str]) -> str | None:
    try:
        tokens = split_command(request)
    except ValueError:
        return None
    return next(
        (
            token.rstrip(".,;:!?")
            for token in tokens
            if Path(token.rstrip(".,;:!?")).suffix.lower() in suffixes
        ),
        None,
    )


def _research_response_guidance(mode: str) -> str:
    if mode == "local":
        scope = (
            "Use only the supplied local-document excerpts. Do not use web "
            "information or unsupported general knowledge; if the excerpts "
            "are insufficient, say so."
        )
    elif mode == "both":
        scope = (
            "Use both local-document and web excerpts where relevant, and "
            "clearly distinguish which source type supports each finding."
        )
    else:
        scope = (
            "Use only the supplied web-source excerpts for research-based "
            "claims, and do not present unsupported general knowledge as a finding."
        )
    return (
        f"{scope} Cite each research-based factual claim with its numbered [n] "
        "citation. Include a concise answer, key findings, material conflicts "
        "or evidence limitations, and a Sources section. Do not invent sources, "
        "URLs, or dates. Clearly distinguish search snippets from extracted "
        "webpage content."
    )


class TerminalApp:
    """Coordinate environment status, input, commands, and local model inference."""

    def __init__(
        self,
        config: AppConfig,
        renderer: Renderer | None = None,
        input_handler: InputHandler | None = None,
        status_provider: Callable[[], dict[str, Any]] | None = None,
        ollama_client: OllamaClient | None = None,
        chat_service: ChatService | None = None,
        rag_service: RagService | None = None,
        research_service: ResearchService | None = None,
    ) -> None:
        self.config = config
        self.renderer = renderer or Renderer()
        self.workspace = CodingWorkspace(Path.cwd())
        self.input_handler = input_handler or InputHandler(self._prompt)
        self.status_provider = status_provider or (
            lambda: gather_environment(
                ollama_url=config.ollama_url,
                model=config.base_model,
            )
        )
        self.ollama_client = ollama_client or OllamaClient(
            config.ollama_url,
            config.base_model,
        )
        self.chat_service = chat_service or ChatService(config, self.ollama_client)
        self.coding_agent = CodingAgent(self.workspace, self.chat_service)
        self.model_manager = ModelManager(config, self.ollama_client)
        self.rag_service = rag_service or RagService(config)
        query_planner = (
            self.chat_service.plan_research_queries
            if isinstance(self.chat_service, ChatService)
            else None
        )
        self.research_service = research_service or ResearchService(
            config,
            self.rag_service,
            query_planner=query_planner,
        )
        self.commands = CommandHandler(
            config,
            self.status_provider,
            lambda: self.chat_service.message_count,
            self.rag_service,
            self.renderer.show_ingestion_progress,
            self.research_service,
            self.renderer.show_web_stage,
            self.model_manager,
            self.renderer.show_model_pull_progress,
            workspace=self.workspace,
            coding_agent=self.coding_agent,
            chat_service=self.chat_service,
        )

    def _prompt(self) -> str:
        cwd = str(self.workspace.cwd)
        home = str(Path.home())
        try:
            relative_to_home = Path(cwd).relative_to(Path(home))
        except ValueError:
            pass
        else:
            cwd = "~" if not relative_to_home.parts else str(Path("~") / relative_to_home)
        mode = " code" if self.workspace.coding_mode else ""
        return f"lowp{mode} {cwd} > "

    async def run(self) -> int:
        try:
            status = await asyncio.to_thread(self.status_provider)
            self.renderer.show_startup(self.config, status)
            self.renderer.show_workspace(str(self.workspace.cwd))
            while True:
                try:
                    message = await self.input_handler.read_message()
                except (EOFError, KeyboardInterrupt):
                    self.renderer.show_goodbye()
                    return 0

                if not message.strip():
                    continue
                result = await self.commands.execute(message)
                if result is None:
                    if not await self._run_chat(message):
                        return 0
                    continue
                if result.action == "exit":
                    self.renderer.show_goodbye()
                    return 0
                if result.action == "clear":
                    self.renderer.clear()
                elif result.action == "help":
                    self.renderer.show_help(result.payload)
                elif result.action == "status":
                    result.payload["model"] = self.config.base_model
                    self.renderer.show_status(result.payload, self.config.app_version)
                elif result.action == "model":
                    self.renderer.show_model(
                        result.payload["config"], result.payload["status"]
                    )
                elif result.action == "model_overview":
                    self.renderer.show_model_overview(result.payload)
                elif result.action == "model_list":
                    self.renderer.show_models(
                        result.payload["active"], result.payload["models"]
                    )
                elif result.action == "model_info":
                    self.renderer.show_model_info(result.payload)
                elif result.action == "model_switched":
                    self.renderer.show_model_switched(result.payload.name)
                elif result.action == "model_pull_confirmation":
                    self.renderer.show_model_pull_confirmation(result.payload)
                elif result.action == "model_pulled":
                    self.renderer.show_model_pulled(result.payload)
                elif result.action.startswith(("training_", "adapter_")):
                    self.renderer.show_training(result.action, result.payload)
                elif result.action.startswith("pdf_"):
                    self.renderer.show_pdf(result.action, result.payload)
                elif result.action == "doctor":
                    self.renderer.show_doctor(result.payload)
                elif result.action == "router_status":
                    self.renderer.show_router_status(result.payload)
                elif result.action == "router_plan":
                    self.renderer.show_route_decision(
                        result.payload["plan"], result.payload["model"]
                    )
                elif result.action == "router_capabilities":
                    self.renderer.show_router_capabilities(result.payload)
                elif result.action == "generation_settings":
                    self.renderer.show_generation_settings(result.payload)
                elif result.action == "code_status":
                    self.renderer.show_code_status(result.payload)
                elif result.action == "code_help":
                    self.renderer.show_code_help()
                elif result.action == "code_stopped":
                    self.renderer.show_code_stopped(result.payload)
                elif result.action == "permissions":
                    self.renderer.show_permissions(result.payload)
                elif result.action == "pwd":
                    self.renderer.show_path(result.payload)
                elif result.action == "directory_list":
                    self.renderer.show_directory_list(result.payload)
                elif result.action == "directory_changed":
                    self.renderer.show_directory_changed(result.payload)
                elif result.action == "directory_tree":
                    self.renderer.show_directory_tree(result.payload)
                elif result.action == "file_search":
                    self.renderer.show_file_search(result.payload)
                elif result.action == "project_search":
                    self.renderer.show_project_search(result.payload)
                elif result.action == "project_info":
                    self.renderer.show_project_info(result.payload)
                elif result.action == "project_opened":
                    self.renderer.show_directory_changed(result.payload)
                elif result.action == "project_instructions":
                    self.renderer.show_project_instructions(result.payload)
                elif result.action == "project_index_confirm":
                    self.renderer.show_project_index_confirmation(result.payload)
                elif result.action == "file_read":
                    self.renderer.show_file_snapshot(result.payload)
                elif result.action == "file_written":
                    self.renderer.show_file_written(
                        result.payload, self.workspace.last_diff
                    )
                elif result.action == "file_deleted":
                    self.renderer.show_file_deleted(result.payload)
                elif result.action == "file_diff":
                    self.renderer.show_file_diff(result.payload)
                elif result.action == "git_output":
                    self.renderer.show_git_output(result.payload)
                elif result.action == "run_confirmation":
                    self.renderer.show_run_confirmation(result.payload)
                elif result.action == "command_result":
                    self.renderer.show_command_result(result.payload)
                elif result.action == "agent_result":
                    self.renderer.show_agent_outcome(result.payload)
                elif result.action == "init_confirmation":
                    self.renderer.show_init_confirmation(result.payload)
                elif result.action == "settings":
                    self.renderer.show_settings(result.payload)
                elif result.action == "reset":
                    self.chat_service.reset()
                    self.renderer.show_reset()
                elif result.action == "error":
                    self.renderer.show_error(result.payload)
                elif result.action == "knowledge":
                    self.renderer.show_knowledge(result.payload)
                elif result.action == "ingestion":
                    self.renderer.show_ingestion_result(result.payload)
                elif result.action == "folder_ingestion":
                    self.renderer.show_folder_ingestion(
                        result.payload["files"], result.payload["results"]
                    )
                elif result.action == "sources":
                    self.renderer.show_sources(
                        result.payload["sources"],
                        result.payload["total"],
                        result.payload["page"],
                        result.payload["page_size"],
                    )
                elif result.action == "source":
                    self.renderer.show_source_details(result.payload)
                elif result.action == "search":
                    self.renderer.show_search_results(result.payload.chunks)
                elif result.action == "remove_source_confirm":
                    self.renderer.show_remove_confirmation(
                        result.payload["id"], result.payload["filename"]
                    )
                elif result.action == "source_removed":
                    self.renderer.show_source_removed(result.payload.filename)
                elif result.action == "remove_all_sources_confirm":
                    self.renderer.show_remove_all_sources_confirmation(result.payload)
                elif result.action == "all_sources_removed":
                    self.renderer.show_all_sources_removed()
                elif result.action == "reindex":
                    self.renderer.show_ingestion_result(result.payload)
                elif result.action == "rag":
                    self.renderer.show_rag_mode(result.payload["enabled"])
                elif result.action == "reset_knowledge_confirm":
                    self.renderer.show_reset_knowledge_confirmation(result.payload)
                elif result.action == "knowledge_reset":
                    self.renderer.show_knowledge_reset()
                elif result.action in {"web_status", "web_settings"}:
                    self.renderer.show_web_status(result.payload)
                elif result.action == "web_search":
                    self.renderer.show_web_search_results(result.payload)
                elif result.action == "web_page":
                    self.renderer.show_web_page(result.payload)
                elif result.action == "web_saved":
                    self.renderer.show_web_saved(result.payload)
                elif result.action == "web_cache":
                    self.renderer.show_web_cache(result.payload)
                elif result.action == "web_cache_cleared":
                    self.renderer.show_web_cache_cleared(result.payload)
                elif result.action == "web_cache_settings":
                    self.renderer.show_web_status(result.payload)
                elif result.action == "copy_code_usage":
                    self.renderer.show_copy_usage()
                elif result.action == "copy_code":
                    try:
                        self.renderer.copy_code_block(result.payload)
                    except (OSError, RuntimeError, ValueError) as exc:
                        self.renderer.show_error(str(exc))
                    else:
                        self.renderer.show_copy_success(result.payload)
                elif result.action == "research":
                    if not await self._run_research(
                        result.payload["question"],
                        result.payload["mode"],
                        fresh=result.payload.get("fresh", False),
                    ):
                        return 0
        finally:
            try:
                await self.model_manager.close()
            finally:
                try:
                    await self.research_service.close()
                finally:
                    try:
                        await self.rag_service.close()
                    finally:
                        await self.ollama_client.aclose()

    async def _run_chat(self, user_message: str) -> bool:
        self.renderer.show_user_message(user_message)
        metrics: GenerationMetrics | None = None
        citations = ()
        retrieved_context = ""
        response_guidance: str | None = None
        local_evidence = False
        route = (
            self.commands.task_router.plan(user_message)
            if self.config.router.automatic
            else None
        )
        if route is not None:
            self.commands.last_route = route
            self.renderer.show_route_decision(route, self.config.base_model)
        try:
            if route is not None and route.intent.value == "document_summarization":
                try:
                    path = _media_path_from_request(user_message, {".pdf"})
                    if path is None:
                        raise ValueError("Name a PDF file in the active workspace to summarize.")
                    result = await self.commands.execute(
                        f"/pdf summarize {shlex.quote(path)}"
                    )
                    if result is None:
                        raise RuntimeError("The media operation did not produce a result.")
                    if result.action == "error":
                        self.renderer.show_error(result.payload)
                    else:
                        self.renderer.show_pdf(result.action, result.payload)
                except (RuntimeError, ValueError, OSError) as exc:
                    self.renderer.show_error(str(exc))
                return True
            routed_research = route is not None and route.research_mode is not None
            if routed_research:
                try:
                    mode = route.research_mode
                    bundle = await self.research_service.research(
                        user_message,
                        mode=mode,
                        progress=lambda stage: self.renderer.show_research_stage(
                            mode, stage
                        ),
                    )
                    self.renderer.show_research_errors(bundle.errors)
                    retrieved_context = bundle.context
                    citations = bundle.citations
                    local_evidence = mode in {"local", "both"}
                    if not bundle.has_evidence:
                        self.renderer.show_research_no_evidence(mode)
                        return True
                    elif bundle.context:
                        response_guidance = _research_response_guidance(mode)
                        self.renderer.show_research_stage(
                            mode, "Generating final response"
                        )
                except (
                    SearchProviderError,
                    RetrievalError,
                    VectorStoreError,
                    RuntimeError,
                    OSError,
                    ValueError,
                ) as exc:
                    logger.warning("Routed research failed.", exc_info=exc)
                    if route.research_mode == "local":
                        self.renderer.show_rag_unavailable(str(exc))
                    else:
                        self.renderer.show_web_unavailable(str(exc))
                    return True
            else:
                try:
                    if await self.rag_service.has_indexed_sources():
                        self.renderer.show_rag_searching()
                        context = await self.rag_service.context_for_chat(user_message)
                        retrieved_context = context.text
                        citations = context.citations
                        local_evidence = bool(context.text)
                except (RetrievalError, VectorStoreError, RuntimeError, OSError, ValueError) as exc:
                    logger.warning("Local document retrieval failed; continuing without RAG.", exc_info=exc)
                    self.renderer.show_rag_unavailable(str(exc))

            if not routed_research and self.research_service.should_research(user_message):
                web_bundle: ResearchBundle | None = None
                try:
                    web_bundle = await self.research_service.research_web(
                        user_message,
                        citation_offset=len(citations),
                        progress=self.renderer.show_web_stage,
                    )
                    retrieved_context = "\n\n".join(
                        part
                        for part in (retrieved_context, web_bundle.context)
                        if part
                    )
                    citations = (*citations, *web_bundle.citations)
                    self.renderer.show_research_errors(web_bundle.errors)
                except (SearchProviderError, RuntimeError, OSError, ValueError) as exc:
                    logger.warning("Automatic web research failed.", exc_info=exc)
                    self.renderer.show_web_unavailable(str(exc))
                if web_bundle is not None and web_bundle.context:
                    self.renderer.show_web_stage("Generating final response")
                    response_guidance = _research_response_guidance(
                        "both" if local_evidence else "web"
                    )

            if self.workspace.coding_mode:
                outcome = await self.coding_agent.start(
                    user_message,
                    retrieved_context=retrieved_context,
                )
                self.renderer.show_agent_outcome(outcome)
                self.renderer.show_citations(citations)
                return True

            self.renderer.begin_stream()
            response = (
                self.chat_service.stream_response(
                    user_message,
                    retrieved_context=retrieved_context,
                    response_guidance=response_guidance,
                )
                if retrieved_context
                else self.chat_service.stream_response(user_message)
            )
            async for event in response:
                if event.content:
                    self.renderer.stream_token(event.content)
                if event.done:
                    metrics = event.metrics
            self.renderer.finish_stream(metrics)
            self.renderer.show_citations(citations)
            self.renderer.show_context_pruned(
                self.chat_service.conversation.last_pruned_messages
            )
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
            if self.workspace.coding_mode:
                self.coding_agent.stop()
            self.renderer.show_generation_stopped()
        except OllamaError as exc:
            self.renderer.end_stream()
            self.renderer.show_error(str(exc))
        except (OSError, ValueError) as exc:
            self.renderer.end_stream()
            self.renderer.show_error(f"Could not complete the local model request: {exc}")
        return True

    async def _run_research(
        self, question: str, mode: str, *, fresh: bool = False
    ) -> bool:
        self.renderer.show_user_message(question)
        show_stage = lambda stage: self.renderer.show_research_stage(mode, stage)
        try:
            bundle = await self.research_service.research(
                question, mode=mode, fresh=fresh, progress=show_stage
            )
        except (SearchProviderError, RuntimeError, OSError, ValueError) as exc:
            self.renderer.show_error(str(exc))
            return True
        self.renderer.show_research_errors(bundle.errors)
        if not bundle.has_evidence:
            self.renderer.show_research_no_evidence(mode)
            return True

        show_stage("Generating final response")
        self.renderer.begin_stream()
        metrics: GenerationMetrics | None = None
        try:
            async for event in self.chat_service.stream_response(
                question,
                retrieved_context=bundle.context,
                response_guidance=_research_response_guidance(mode),
            ):
                if event.content:
                    self.renderer.stream_token(event.content)
                if event.done:
                    metrics = event.metrics
            self.renderer.finish_stream(metrics)
            self.renderer.show_citations(bundle.citations)
            self.renderer.show_context_pruned(
                self.chat_service.conversation.last_pruned_messages
            )
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None:
                current_task.uncancel()
            self.renderer.show_generation_stopped()
        except OllamaError as exc:
            self.renderer.end_stream()
            self.renderer.show_error(str(exc))
        except (OSError, ValueError) as exc:
            self.renderer.end_stream()
            self.renderer.show_error(
                f"Could not complete the local model request: {exc}"
            )
        return True
