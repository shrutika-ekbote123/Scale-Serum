"""Who sees which tab, and whose rep row.

Roles in scrumdb are custom per company ("Sales Exec", "Editor", "Testing 9-"),
so access is decided from each role's PERMISSION list, which the main app
forwards with the request:

    X-User-Id           users.id of the viewer
    X-User-Role         users.role ("superAdmin" | "user")
    X-User-Permissions  the role's permissions, comma-separated

The main app is the trusted caller (it holds the API key), so a request with no
viewer headers is a service call and sees everything - the same as before this
module existed. A viewer who sends headers is filtered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SECTION_PERMISSIONS = {
    "sales": {"sales_calls", "crm"},
    "ads": {"meta_ads", "google_ads", "linkedin_ads", "all_accounts"},
    "whatsapp": {"whatsapp_business"},
    "leads": {"crm", "lead_search", "lead_journey", "lead_analytics"},
}
PAGE_PERMISSION = "briefings"
TEAM_PERMISSIONS = {"users_roles", "briefings_team"}
SUPER_ROLES = {"superadmin"}


@dataclass(frozen=True)
class Viewer:
    user_id: Optional[str]
    role: Optional[str]
    permissions: frozenset
    service: bool          # no viewer headers: the calling service itself

    @property
    def is_super(self) -> bool:
        return self.service or (self.role or "").strip().lower() in SUPER_ROLES

    @property
    def can_open_page(self) -> bool:
        return self.is_super or PAGE_PERMISSION in self.permissions

    @property
    def team_view(self) -> bool:
        return self.is_super or bool(TEAM_PERMISSIONS & self.permissions)

    def sections(self) -> list[str]:
        """The data tabs this viewer may see, in display order."""
        if not self.can_open_page:
            return []
        if self.is_super:
            return list(SECTION_PERMISSIONS)
        return [s for s, perms in SECTION_PERMISSIONS.items() if perms & self.permissions]

    @property
    def full_access(self) -> bool:
        return len(self.sections()) == len(SECTION_PERMISSIONS)


def viewer_from(user_id: Optional[str], role: Optional[str],
                permissions: Optional[str]) -> Viewer:
    if user_id is None and role is None and permissions is None:
        return Viewer(None, None, frozenset(), service=True)
    perms = frozenset(p.strip().lower() for p in (permissions or "").split(",") if p.strip())
    return Viewer(user_id, role, perms, service=False)
