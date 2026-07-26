use dynamo_rs::artifacts::{flat_run_mask, flat_run_mask_with_tolerance};

#[test]
fn tolerance_helper_is_public_without_changing_strict_wrapper() {
    let data = [0.0, 0.004, -0.003, 0.002];

    assert_eq!(flat_run_mask(&data, 4), vec![false; 4]);
    let tolerant = flat_run_mask_with_tolerance(&data, 4, 0.01);
    assert_eq!(tolerant, vec![true; 4]);
}
