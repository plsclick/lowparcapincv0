# Self-hosted SearXNG on Linux

LowP's only web-search provider is SearXNG's HTTP JSON search API. SearXNG is
an independent service. `./install.sh` first checks the default local JSON
endpoint and reuses a healthy instance without changing it. Otherwise, when an
already installed Docker or Podman runtime is operational, the installer
offers to create a LowP-owned container and settings directory. It never
installs a container runtime, requires root, takes over unowned containers, or
publishes the service beyond loopback. Manual setup remains available below.

## Requirements and local-only binding

Use Docker Engine or Podman. Follow the container runtime's official
installation instructions for your Linux distribution. LowP itself does not
install the runtime and works without SearXNG; web commands clearly report
that search is unavailable when there is no instance. If neither runtime is
available, `./install.sh` reports SearXNG as not ready and asks you to enable
one before rerunning the installer.

The published host port must bind to loopback, not `0.0.0.0`. For Docker
Compose, the relevant service configuration is:

```yaml
services:
  searxng:
    image: searxng/searxng@sha256:c750ee1d859b26d3aeae51c07f6a48c3a0bfbaa9195d71a18b258862d88c4c3c
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml:rw,Z
    environment:
      - SEARXNG_BASE_URL=http://127.0.0.1:8080/
    restart: unless-stopped
```

Create a new, private directory for this deployment and add the Compose file
and settings file there using your editor. Do not overwrite any existing
deployment files. The settings file is writable because the SearXNG container
entrypoint prepares it during startup. `:Z` applies a private SELinux label,
which is needed on SELinux-enabled Linux distributions such as Fedora. Keep the
SearXNG secret key private. Generate one locally:

```bash
openssl rand -hex 32
```

## Enable JSON and configure the instance

SearXNG does not necessarily enable JSON output by default. In the instance's
`settings.yml`, ensure the search formats include both `html` and `json`. A
minimal example (merge settings into an existing file rather than replacing
unrelated existing values) is:

```yaml
use_default_settings: true

general:
  instance_name: "Local LowP Search"

search:
  safe_search: 1
  default_lang: "en"
  formats:
    - html
    - json

server:
  secret_key: "REPLACE_WITH_A_RANDOM_PRIVATE_VALUE"
  limiter: true
  public_instance: false
```

Replace the placeholder secret locally. Do not commit the real key. SearXNG's
available engines and category names depend on its configuration and upstream
availability. Keep the default engine set unless you have a reason to change
it; LowP's optional categories and language settings are sent to the instance.
Safe search values are `0` (off), `1` (moderate), and `2` (strict). Configure
reasonable request limits for your use and do not expose an unauthenticated
public instance.

Consult the current [SearXNG documentation](https://docs.searxng.org/) for
configuration options and [search API documentation](https://docs.searxng.org/dev/search_api.html)
for the JSON endpoint. LowP requests `GET /search` with `q`, `format=json`,
`pageno`, and optional `categories`, `language`, and `safesearch` parameters.
The response must contain a `results` array; individual missing optional
fields are tolerated.

## Start, stop, update and test

The automated installer uses a pinned SearXNG image digest and stores its
configuration under `$XDG_DATA_HOME/lowp/searxng/` (default
`~/.local/share/lowp/searxng/`). It applies a LowP ownership label to its
container. Healthy existing services are reused; an unmarked container named
`lowp-searxng` is left untouched and reported as a conflict. The image download
requires confirmation. The generated settings enable JSON and keep the
published host port bound to `127.0.0.1`.

For a manually managed Compose deployment, from the directory where you
created your Compose file:

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 searxng
curl 'http://127.0.0.1:8080/search?q=lowp&format=json'
docker compose stop
```

Verify actual JSON results (not only HTTP status) with:

```bash
python3 -c 'import json,urllib.request; p=json.load(urllib.request.urlopen("http://127.0.0.1:8080/search?q=test&format=json", timeout=10)); assert isinstance(p.get("results"), list) and any(isinstance(r, dict) and r.get("title") and r.get("url") for r in p["results"]); print("SearXNG JSON search verified")'
```

For rootless Podman, use `podman compose` if a Compose provider is installed,
or run the equivalent Podman-managed container with the same loopback-only
port mapping. Do not use `sudo` unless required by your own container runtime
installation. To update, review the SearXNG release notes, then pull and
recreate the container using your chosen runtime; preserve and back up your
settings before updating. These commands do not remove your volumes or
configuration.

## Connect LowP

LowP defaults to `http://127.0.0.1:8080`. To use another instance, configure
the URL in the shell or through `/web-settings`:

```bash
export SEARXNG_BASE_URL=http://127.0.0.1:8080
lowp
```

Or in LowP:

```text
/web-settings base_url http://127.0.0.1:8080
/web
/web-search local search test
```

HTTP is accepted only for loopback. A remote instance must use HTTPS. By
choosing a remote endpoint you trust its operator with your search queries and
connection metadata.

## Troubleshooting

- **Connection refused:** verify the container is running and the host mapping
  is exactly `127.0.0.1:8080:8080`; inspect its logs.
- **Permission denied for `/etc/searxng/settings.yml` on Fedora:** use the
  `:rw,Z` bind mount shown above, then recreate the container. `:Z` labels the
  file for this container and `rw` lets its startup entrypoint prepare the
  settings file.
- **HTTP 403 on `/search?...&format=json`:** check that `json` appears in
  `search.formats`, then restart the SearXNG service.
- **No results:** inspect enabled engines, the requested category/language,
  safe-search, and upstream engine health in SearXNG.
- **LowP rejects the URL:** use a loopback HTTP address for local use, or a
  valid HTTPS URL for a remote instance. Credentials, query strings, and
  fragments are not accepted in the base URL.
- **Container is reachable from another machine:** stop it and correct the
  published host address to `127.0.0.1`; do not publish it publicly without
  deliberate authentication and network controls.

SearXNG aggregates third-party engines and is subject to their availability,
rate limits, and access policies. Neither SearXNG nor LowP guarantees unlimited
search access.

## Uninstallation and upgrades

Rerunning `./install.sh` updates the LowP package without replacing user
configuration, databases, caches, datasets, or adapters. A healthy local
SearXNG service is reused. The LowP uninstaller preserves SearXNG by default;
it offers a separate confirmation to remove only a container bearing the
LowP ownership label and a settings directory bearing LowP's marker. Other
containers, manually managed deployments, model downloads, and user data are
not removed.
