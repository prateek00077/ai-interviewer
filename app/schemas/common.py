"""Shared response shapes.

One pagination envelope for every list endpoint. Keyset pagination would scale
better, but every list here is scoped to a single tenant's own rows -- an org
with enough candidates for OFFSET to hurt is a problem worth having, and a
limit/offset API is what a recruiter UI's page controls actually want.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


class Page(BaseModel, Generic[T]):
    items: list[T]
    # The count BEFORE limit/offset, so a client can render "page 2 of 7".
    total: int
    limit: int
    offset: int
