from football_predictor.world_cup_live import _espn_starters_from_payload


def test_espn_starters_are_extracted_only_when_marked():
    payload = {
        "rosters": [
            {
                "team": {"displayName": "Brazil"},
                "roster": [
                    {"starter": True, "athlete": {"displayName": f"Brazil Player {i}"}}
                    for i in range(1, 12)
                ] + [
                    {"starter": False, "athlete": {"displayName": "Brazil Substitute"}}
                ],
            },
            {
                "team": {"displayName": "Morocco"},
                "roster": [
                    {"starter": True, "athlete": {"displayName": f"Morocco Player {i}"}}
                    for i in range(1, 12)
                ],
            },
        ]
    }
    brazil = _espn_starters_from_payload(payload, "Brazil")
    morocco = _espn_starters_from_payload(payload, "Morocco")
    assert len(brazil) == 11
    assert len(morocco) == 11
    assert "Brazil Substitute" not in brazil


def test_espn_squad_without_starter_flags_is_not_treated_as_lineup():
    payload = {
        "rosters": [
            {
                "team": {"displayName": "Brazil"},
                "roster": [
                    {"athlete": {"displayName": f"Brazil Player {i}"}}
                    for i in range(1, 27)
                ],
            }
        ]
    }
    assert _espn_starters_from_payload(payload, "Brazil") == []
