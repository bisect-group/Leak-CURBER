from __future__ import annotations


def test_model_name_carries_every_setting() -> None:
    from src.data.components.embedders.drfp import drfp_model_name

    base = drfp_model_name(3, 0, 2048, False, False)
    assert base == "radius3_min0_len2048_rings0_h0"
    assert (
        len(
            {
                base,
                drfp_model_name(2, 0, 2048, False, False),
                drfp_model_name(3, 0, 1024, False, False),
                drfp_model_name(3, 0, 2048, True, False),
                drfp_model_name(3, 0, 2048, False, True),
            }
        )
        == 5
    )
