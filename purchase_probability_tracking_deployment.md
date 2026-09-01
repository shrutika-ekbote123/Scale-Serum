# Purchase Probability — Tracking Deployment (Track 2)

**DOCUMENTATION ONLY. NOTHING EXECUTED.**

No tracking change was made. No key, allowlist, snippet, schema or row was modified. This
document records the steps and the open questions so the fix can be authorised separately.

**Track 2 is independent of Track 1.** The production MVP ships without behavioural data and
must not be modified to compensate for its absence.

---

## 1. Current problem

| Fact | Value |
|---|---|
| `tracking_events` | **0 rows** |
| `tracking_visitors` | **0 rows** |
| `tracking_brand_keys` | 5, all `status='active'`, none revoked |
| `first_event_at` / `last_event_at` | **NULL on all five keys** |
| `tracking_events` lifetime inserts (`n_tup_ins`) | **0** |
| Collector `https://app.scaleserum.com/ss.js` | HTTP 200, 27,184 bytes, v1.0.0 — deployed |
| Migration `20260716120000-create-tracking-module.js` | applied |

Two independent proofs that no event has **ever** been ingested: PostgreSQL's cumulative
insert counter has never incremented (`n_tup_del = 0` too, so this is not deletion), and
`first_event_at` — application-maintained, survives statistics resets — is NULL on every key.

## 2. Root cause

> The tracking module was built, migrated and exercised in a demo/QA capacity — write keys
> were minted from the admin UI — but **the collector snippet was never installed on any live
> customer site.** The pipeline has never completed end-to-end.

Supporting evidence, strongest first:

1. `first_event_at` NULL on all 5 keys.
2. `n_tup_ins = 0`.
3. Three of five keys are unmistakably test data.
4. The two brands with real volume have `allowed_domains = []` or a reserved example domain.
5. `ss.js` returns silently when no write key resolves — its own comment: *"a snippet pasted
   without its key looks installed and tracks nothing."*
6. Customer forms are Wix, where snippet installation is a manual step separate from key
   generation.

## 3. Brand / key / domain mapping (write keys masked)

| Brand | Created | `allowed_domains` | Leads | Classification |
|---|---|---|---|---|
| **Lawtorney** | 2026-07-16 06:45 | **`[]`** | **20,954** | **B — missing domain configuration** |
| Acme | 2026-07-16 09:04 | `[]` | 0 | C — test/placeholder |
| Kaizen CRM | 2026-07-20 06:53 | `["kaizencrm.example"]` | 7,770 | C — `.example` is an RFC 2606 reserved TLD |
| DI | 2026-08-19 06:55 | `["scaleserum.com"]` | 1 | C — points at ScaleSerum's own site |
| Lawttorney | 2026-08-25 07:32 | `["lawttorney.com"]` | 0 | D — typo duplicate of "Lawtorney" |

**No key is production-ready.** All five share the default `script_url`
`https://app.scaleserum.com/ss.js` and `collector_host` `https://app.scaleserum.com`.

**Target brand: Lawtorney** — 20,954 of 28,740 leads (~73%), and ~97% of the leads the model
was trained on.

**Production domain: UNKNOWN — do not guess.** Confirmed facts: the brand is named
"Lawtorney"; a *different* brand row carries `lawttorney.com`. Whether the real site is
`lawtorney.com`, `lawttorney.com`, or something else **cannot be established from the
database or from any file in this workspace**. It must be confirmed from the customer's
actual site before anything is configured. `brands.website` should be checked first.

## 4. Blocking open question — `allowed_domains = []`

The `/collect` handler belongs to the Node/Sequelize service, which is **not in this
workspace**. Therefore:

> **Unresolved: whether `allowed_domains = []` means "allow all" or "deny all".**

The column is `NOT NULL DEFAULT '[]'::json`, which suggests it is meant to be populated. If
it denies all, then Lawtorney's key would reject every request even after a correct snippet
install — making this a prerequisite, not a follow-up.

**This must be settled by reading the handler before any install.** It was not probed:
`ss.js` falls back to `img.src = endpoint + "?d=" + encoded`, so `/collect` accepts events
over GET and any probe risks inserting a row.

Also unresolved from here: whether domain matching is exact or subdomain/wildcard, what
happens on rejection, and whether GET and POST differ. There is **no dead-letter table
anywhere in the schema**, so rejected events leave no trace — a diagnosability gap worth
closing regardless.

## 5. How `ss.js` receives its key

```js
writeKey = window.SSKey
        || document.currentScript.getAttribute('data-key')
        || param('k', src)
        || null;

host     = window.SSHost || currentScript.getAttribute('data-host') || origin-of-script-src;
endpoint = host.replace(/\/+$/, '') + '/collect';
```

If no key resolves, `boot()` warns to console and returns. The page looks instrumented and
sends nothing.

## 6. Snippet configuration

```html
<!-- head, every page -->
<script async
        src="https://app.scaleserum.com/ss.js"
        data-key="ssk_…"></script>
```

Use the **Lawtorney** key. Do not paste a key belonging to another brand — `visitor_id` is
only unique within a brand, and cross-brand contamination is unrecoverable after the fact.

## 7. Wix installation location

Wix Dashboard → **Settings → Custom Code** → *Add Custom Code* →
place in **Head**, load on **All pages**, category **Essential/Functional**.

Wix strips or defers some third-party scripts depending on consent settings; verify the tag
is actually present in the served HTML, not merely saved in the dashboard.

## 8. Verification procedure

1. **Client** — load the site with `?ss_debug=1`. The console should log
   `booted {key, endpoint, visitor, session}`. No log ⇒ the snippet is not executing or the
   key did not resolve.
2. **Network** — confirm a request to `https://app.scaleserum.com/collect` in DevTools.
   Note the transport: `sendBeacon` requests appear under a different type than `fetch`.
3. **Key stamp** (cheapest server-side check):
   ```sql
   SELECT first_event_at, last_event_at FROM tracking_brand_keys WHERE brand_id = '<lawtorney>';
   ```
   `first_event_at` becoming non-NULL is the single clearest success signal.
4. **Rows**:
   ```sql
   SELECT count(*), min(received_at), max(received_at) FROM tracking_events WHERE brand_id = '<lawtorney>';
   ```
5. **Visitors** — `tracking_visitors` should populate with unique `(brand_id, visitor_id)`;
   confirm the same `visitor_id` recurs across pages rather than a new one per page view
   (that would indicate cookie/localStorage failure).
6. **Stitching** — after a form submission, check `tracking_visitors.lead_id` populates and
   that `lead_identities` gains `anon_cookie` rows. **Currently 0 of each**, and
   `touchpoint_events.anon_id` is empty on all 27,135 rows, so both ends of the bridge are
   presently unbuilt. This must be verified empirically, not assumed from schema.

## 9. Expected first-event behaviour

A `page_view` should arrive within seconds of the first real page load. `received_at` and
`created_at` are server-side and `NOT NULL`; `client_ts` is browser-supplied and nullable.

## 10. Security considerations

- **Write keys are public by design** — they sit in page source. `allowed_domains` is the
  only thing preventing another site from posting events against a brand. Configure it.
- `ss.js` sends `credentials: 'include'`; confirm CORS and cookie settings are intentional.
- `tracking_events` stores `ip`, `user_agent` and `country`. Confirm the privacy/consent
  position before using them as features.
- Prefer a first-party CNAME (`t.<domain>`) later — the script already supports it via
  `data-host`, and it stops Safari/ITP truncating the cookie after 7 days.

## 11. Rollback

Set the key's `status` to revoked (or clear `allowed_domains`) and remove the snippet from
Wix Custom Code. No schema change is involved, so rollback is configuration-only. Any events
already collected remain; delete them only with an explicit, separately authorised decision.

## 12. Data maturation before Experiment 3

Derived from observed velocity: 14,692 leads / 160 positives over 69 days =
**213 leads/day, 2.32 positives/day**.

| Component | Requirement |
|---|---|
| Pre-t₀ event accumulation | **Cannot be established yet** — browse-to-submit latency is unknown because no visit history exists. Measurable ~1–2 weeks after events start. |
| Outcome-label maturation | **21 days** (V3 target window; 94% of 30-day conversions land inside it) |
| V1-equivalent sample (~100 OOF positives) | ~43 days of leads |
| **Total to a first look** | **≈ 64 days** |
| **Total to a decisive answer** | **≈ 4–6 months** (200–400 positives) |

Leads arriving before install have **no** pre-t₀ behavioural history and are permanently
unusable for behavioural features. The clock starts at install.

## 13. Future behavioural features (inventory only — do not build)

Page views before t₀ · unique sessions · session duration · pages per session · pricing-page
visits · repeat visits · visit-to-form latency · referrer and UTM (which would finally fill
the gap where `touchpoint_events.utm_campaign` is populated on 1 of 27,135 rows) ·
engagement depth · anonymous-to-known history.

All must be computed from the **immutable `tracking_events` log under an as-of-t₀ cutoff**:
`created_at <= t0 AND received_at <= t0`, with `received_at − client_ts <= 1h` as a
backdating detector. **`tracking_visitors` is current state** (`last_seen_at`, `last_touch`
mutate in place) and must never be a feature source — that is the same defect class as
`leads.last_activity_at`.

Re-run the write-lag test (`created_at − occurred_at`) on `tracking_events` as soon as rows
exist. The `ad_click`/`call` backfill was invisible until exactly that test was applied.

## 14. Required before authorisation

1. **Read the `/collect` handler** — settle the `allowed_domains = []` semantics. Blocking.
2. **Confirm Lawtorney's real production domain** from the customer or `brands.website`. Blocking.
3. Decide whether to retire the Acme / DI demo keys and the Lawttorney typo duplicate.
4. Decide whether to add rejected-event logging before or after go-live.

## 15. What must NOT change

No dummy data, no fabricated `tracking_events`, no backfill of historical browsing — there
is no history to recover, and synthesising it would reproduce exactly the backdating defect
that corrupted `ad_click` and `call`. No DDL on the tracking tables; the schema is correct.
No modification to V1, V2, V3, the ML specification, or the Track 1 production model.
