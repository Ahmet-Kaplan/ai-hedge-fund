"""Errors shared by the free data sources."""


class DataSourceError(Exception):
    """A free data source failed for infrastructure reasons (HTTP error, network).

    Like FDClientError, distinct from "no data exists" — callers must not treat
    it as an empty answer.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
