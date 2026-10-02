# Runtime image for both the scheduled paper-trading job and the dashboard.
#
# One image serves both: the job runs the `aihf` entrypoint below, and the
# dashboard overrides the command to launch uvicorn. Two images would mean two
# builds of the same package and a chance for them to drift apart.
#
# Python is pinned to 3.12 because pyproject caps numpy below 2.0, and numpy
# 1.x publishes no wheels for 3.13+. Moving this to a newer tag will fail the
# build, not fall back gracefully.
FROM python:3.12-slim AS build

WORKDIR /src
# Only what the build backend needs to resolve and install the package.
COPY pyproject.toml README.md ./
COPY hedge_fund ./hedge_fund

# [azure] pulls in azure-identity, so the job authenticates to Azure OpenAI
# with its managed identity instead of a long-lived key. [web] adds the
# dashboard's server; the CLI does not import it.
RUN pip install --no-cache-dir --prefix=/install ".[azure,web]"


FROM python:3.12-slim

COPY --from=build /install /usr/local

# HOME drives hedge_fund/paths.py:23 (USER_DIR = Path.home() / ".hedge-fund"),
# which is not otherwise configurable. Pointing HOME at the mounted volume is
# what makes the ledger survive a container restart, with no code change.
ENV HOME=/data \
    PYTHONUNBUFFERED=1

# Fixed uid so the Azure Files mount's ownership is predictable.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin aihf \
    && mkdir -p /data \
    && chown 10001:10001 /data
USER 10001

WORKDIR /data

# No CMD with a fund name: the job supplies it, so one image serves every fund.
ENTRYPOINT ["aihf"]
