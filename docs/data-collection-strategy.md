# LLMART Data Collection Strategy

## 1. Overview

This document defines the data sourcing, revenue estimation, labeling, and scaling
plan for the LLMART game selection pipeline. The goal is to build a reliable
training dataset of indie/AA games released on Steam (2018-2024) with
ground-truth revenue labels.

---

## 2. Data Sources

### 2.1 Steam Store API (Primary)

| Item | Detail |
|---|---|
| **Data available** | App details (name, developer, publisher, price, tags, genres, release date, Metacritic score), review summary (total reviews, positive ratio), player achievements |
| **Access** | Public REST API `https://store.steampowered.com/api/appdetails?appids={id}` |
| **Rate limits** | ~200 requests/5 min per IP (undocumented, ~1 req/1.5s safe) |
| **Cost** | Free |
| **Auth** | None required; optional Steam Web API key for extended endpoints |
| **Notes** | Does not expose sales/revenue. Review count is the primary proxy. Some apps may be region-restricted. |

### 2.2 SteamSpy (Secondary)

| Item | Detail |
|---|---|
| **Data available** | Estimated owners (range buckets), average/median playtime, CCU, tags with vote counts, price history |
| **Access** | REST API `https://steamspy.com/api.php?request=appdetails&appid={id}` |
| **Rate limits** | 4 requests/second |
| **Cost** | Free tier (basic); Patreon ($1-5/mo) for full historical data |
| **Auth** | None for basic; Patreon key for premium |
| **Notes** | Owner estimates became less precise after Steam privacy changes (2018). Still useful for relative comparison. |

### 2.3 VG Insights (Enrichment)

| Item | Detail |
|---|---|
| **Data available** | Estimated revenue, estimated units sold, follower count, wishlist estimates, genre benchmarks |
| **Access** | Web scraping or paid API |
| **Rate limits** | N/A (web) or per subscription |
| **Cost** | Free (limited); Pro $9.99/mo |
| **Auth** | Account required for Pro |
| **Notes** | Revenue estimates use proprietary model. Good for cross-validation. |

### 2.4 Gamalytic (Enrichment)

| Item | Detail |
|---|---|
| **Data available** | Revenue estimates, player count estimates, genre analytics, publisher analytics |
| **Access** | Web dashboard + CSV export |
| **Rate limits** | Per subscription plan |
| **Cost** | Free (limited); paid plans from ~$15/mo |
| **Auth** | Account required |
| **Notes** | Focuses on Steam market analytics. Useful for genre-level benchmarks. |

### 2.5 SteamDB (Reference / Validation)

| Item | Detail |
|---|---|
| **Data available** | Historical player counts, price history, update history, sub/package details, achievement stats, follower history |
| **Access** | Web only (no public API). Manual lookup or community tools. |
| **Rate limits** | N/A (manual) |
| **Cost** | Free |
| **Auth** | None |
| **Notes** | Best source for historical CCU data. No direct scraping API. Use for manual validation of top titles. |

---

## 3. Revenue Estimation: Boxleiter Method

### 3.1 Formula

```
Y1_Revenue = review_count * review_multiplier * price_usd * steam_net_share
```

Where:
- `review_multiplier = 30` (Boxleiter consensus for 2020-2024 era)
- `steam_net_share = 0.70` (after Steam's 30% cut)

### 3.2 Assumptions & Caveats

| Factor | Value | Notes |
|---|---|---|
| Review multiplier | 30 | Conservative mid-range. Literature: 20-50. Chinese-market games may skew higher (40-60). |
| Steam revenue share | 70% | Standard. Drops to 75%/80% at $10M/$50M thresholds (ignored for indie scale). |
| Refund adjustment | Not applied | Boxleiter multiplier implicitly accounts for refunds. |
| DLC / MTX | Not included | Y1 estimate is base game only. |
| Bundle sales | Partially included | Reviews from bundle buyers are counted. Lower effective price is not adjusted. |

### 3.3 Cross-Validation

For Phase 1 (n=15), manually cross-check against:
1. VG Insights revenue estimate (if available)
2. SteamSpy owner ranges * price * 0.70
3. Developer/publisher disclosures (GDC talks, press releases)
4. Gamalytic estimates

Accept if 3+ sources agree within 40% range.

---

## 4. Q1 Revenue Share

Q1 (first 90 days) revenue as a fraction of Y1 revenue depends on the game's
monetization and engagement pattern:

| Type | Q1/Y1 Ratio | Typical Genres | Rationale |
|---|---|---|---|
| **Front-loaded** | ~8% of Y1 (or ~60-70% of Y1 in Q1) | Premium single-player, horror co-op, narrative | Launch spike with rapid decay. Most revenue in first month. |
| **Balanced** | ~6.5% of Y1 (or ~50% of Y1 in Q1) | Roguelikes, deckbuilders, action RPGs | Steady post-launch with moderate long tail. |
| **Live-service** | ~5% of Y1 (or ~35-45% of Y1 in Q1) | Survival craft, city builders, multiplayer sandbox | Slow build with updates driving sustained revenue. |

**Clarification on ratios**: The Q1 share values above express Q1 as a
proportion of total Y1. For Boxleiter-estimated Y1 revenue, the Q1 estimate is:

```
Q1_Revenue = Y1_Revenue * q1_share
```

Where `q1_share`:
- Front-loaded: 0.65
- Balanced: 0.50
- Live-service: 0.40

---

## 5. Hit-Tier Labeling (4-Tier Ground Truth)

Based on estimated Q1 revenue:

| Tier | Q1 Revenue Range | Target % in Dataset | Example Games |
|---|---|---|---|
| **Mega** | >= $20M | ~5% (3 games) | Balatro, Lethal Company, Palworld |
| **Hit** | $2M - $20M | ~15% (9 games) | Cult of the Lamb, Inscryption, Slay the Spire |
| **Side** | $250K - $2M | ~20% (12 games) | Dome Keeper, Core Keeper, Halls of Torment |
| **Hobby** | < $250K | ~60% (36 games) | Patch Quest, small indie titles |

### Mapping to Signal

| Tier | Expected Signal | Rationale |
|---|---|---|
| Mega | GREEN | Q1 P25 >> $250K |
| Hit | GREEN | Q1 P25 >= $250K typically |
| Side | YELLOW | P25 < $250K but P50 >= $250K |
| Hobby | RED | P50 < $250K |

---

## 6. Sample Size Roadmap

| Phase | n (games) | Purpose | Data Quality |
|---|---|---|---|
| **Phase 0 (Demo)** | 8 | CLI demo, formula validation | Manual curation |
| **Phase 1 (Pilot)** | >= 15 | Initial LambdaMART training, sanity check | Cross-validated revenue |
| **Phase 2 (Alpha)** | >= 50 | Tier distribution validation, NDCG@30 tuning | Boxleiter + 1 secondary source |
| **Phase 3 (Production)** | >= 200 | Full pipeline, ECE calibration, drift detection | Boxleiter + 2 secondary sources |

### Phase 2 Target Breakdown (n=60)

| Tier | Count | % |
|---|---|---|
| Mega | 3 | 5% |
| Hit | 9 | 15% |
| Side | 12 | 20% |
| Hobby | 36 | 60% |

---

## 7. Data Collection Pipeline

### Step 1: Seed List Construction
1. Curate list of 200+ indie games from 2021-2024 via SteamSpy top lists
2. Filter: indie/AA only (exclude major publisher titles)
3. Filter: must have >= 50 reviews (minimum signal)

### Step 2: Steam API Scraping
```python
# Pseudocode
for app_id in seed_list:
    details = steam_api.get_app_details(app_id)
    reviews = steam_api.get_review_summary(app_id)
    record = {
        "title": details["name"],
        "genre": classify_genre(details["genres"], details["tags"]),
        "developer": details["developers"][0],
        "steam_rating": reviews["positive"] / reviews["total"],
        "review_count": reviews["total"],
        "price_usd": details["price_overview"]["final"] / 100,
        "release_year": parse_year(details["release_date"]),
        "tags": top_5_tags(details["tags"]),
    }
    time.sleep(1.5)  # respect rate limit
```

### Step 3: Revenue Estimation
```python
record["estimated_y1_revenue"] = (
    record["review_count"] * 30 * record["price_usd"] * 0.70
)
q1_share = {"front-loaded": 0.65, "balanced": 0.50, "live-service": 0.40}
record["estimated_q1_revenue"] = (
    record["estimated_y1_revenue"] * q1_share[record["genre_q1_type"]]
)
```

### Step 4: Tier Assignment
```python
q1 = record["estimated_q1_revenue"]
if q1 >= 20_000_000:
    record["hit_tier"] = "Mega"
elif q1 >= 2_000_000:
    record["hit_tier"] = "Hit"
elif q1 >= 250_000:
    record["hit_tier"] = "Side"
else:
    record["hit_tier"] = "Hobby"
```

### Step 5: Developer History Enrichment
```python
dev_games = steam_api.search_by_developer(record["developer"])
record["developer_games_released"] = len(dev_games)
record["developer_successes"] = count_successes(dev_games, threshold=500_reviews)
```

### Step 6: Cross-Validation (Phase 2+)
- Compare Boxleiter estimate with VG Insights
- Flag discrepancies > 2x for manual review
- Maintain `confidence_score` field (0.0-1.0)

---

## 8. Data Quality Checks

| Check | Threshold | Action |
|---|---|---|
| Missing fields | 0 nulls in required fields | Reject record |
| Review count plausibility | > 50 | Reject if too low for signal |
| Price range | $0.99 - $69.99 | Flag F2P or premium outliers |
| Rating range | 0.30 - 1.00 | Flag if < 0.30 (likely review-bombed) |
| Revenue cross-validation | Within 2x of secondary source | Flag for manual review |
| Genre classification | Must map to genre_params.json | Manual reclassify if unmapped |
| Tier distribution | Within 10% of target | Resample if skewed |

---

## 9. Future Enhancements

1. **Automated scraping pipeline** using Prefect/Airflow for monthly updates
2. **Wishlist data** from SteamDB for pre-launch prediction
3. **CCU time series** for live-service engagement modeling
4. **Metacritic/OpenCritic** scores as additional features
5. **Social media signals** (Reddit, Twitter/X mentions) for virality scoring
6. **Regional pricing** adjustments for non-USD markets
