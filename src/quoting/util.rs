use crate::math::uint::U256;
use crate::quoting::types::Tick;

// Function to find the nearest initialized tick index.
pub fn find_nearest_initialized_tick_index(sorted_ticks: &[Tick], tick: i32) -> Option<usize> {
    let mut l = 0usize;
    let mut r = sorted_ticks.len();

    while l < r {
        let mid = (l + r) / 2;
        let mid_tick = sorted_ticks[mid].index;
        if mid_tick <= tick {
            if mid == sorted_ticks.len() - 1 || sorted_ticks[mid + 1].index > tick {
                return Some(mid);
            } else {
                l = mid;
            }
        } else {
            r = mid;
        }
    }

    None
}

const LOG_BASE_SQRT_TICK_SIZE: f64 = 4.9999975000016666654166676666658333340476184226196031741031750577196410537756684185262518589393595459766211405607685305832e-7;

pub fn u256_to_float_base_x128(x128: U256) -> f64 {
    x128.0[0] as f64 / 340282366920938463463374607431768211456f64
        + (x128.0[1] as f64 / 18446744073709551616f64)
        + x128.0[2] as f64
        + (x128.0[3] as f64 * 18446744073709551616f64)
}

pub fn approximate_number_of_tick_spacings_crossed(
    starting_sqrt_ratio: U256,
    ending_sqrt_ratio: U256,
    tick_spacing: u32,
) -> u32 {
    let start: f64 = u256_to_float_base_x128(starting_sqrt_ratio);
    let end: f64 = u256_to_float_base_x128(ending_sqrt_ratio);
    let ticks_crossed = ((start.ln() - end.ln()).abs() / LOG_BASE_SQRT_TICK_SIZE) as u32;
    ticks_crossed / tick_spacing
}

#[cfg(test)]
mod tests {
    use crate::math::tick::{MAX_SQRT_RATIO, MIN_SQRT_RATIO};
    use crate::math::uint::U256;
    use crate::quoting::types::Tick;
    use crate::quoting::util::find_nearest_initialized_tick_index;
    use crate::quoting::util::{
        approximate_number_of_tick_spacings_crossed, u256_to_float_base_x128,
    };
    use alloc::vec;

    #[test]
    fn test_find_nearest_initialized_tick_index_no_ticks() {
        assert_eq!(find_nearest_initialized_tick_index(&vec![], 0), None);
    }

    #[test]
    fn test_find_nearest_initialized_tick_index_one_tick_less_than() {
        assert_eq!(
            find_nearest_initialized_tick_index(
                &vec![Tick {
                    index: -1,
                    liquidity_delta: 1,
                }],
                0
            ),
            Some(0)
        );
    }

    #[test]
    fn test_find_nearest_initialized_tick_index_one_tick_equal_to() {
        assert_eq!(
            find_nearest_initialized_tick_index(
                &vec![Tick {
                    index: 0,
                    liquidity_delta: 1,
                }],
                0
            ),
            Some(0)
        );
    }

    #[test]
    fn test_find_nearest_initialized_tick_index_one_tick_greater_than() {
        assert_eq!(
            find_nearest_initialized_tick_index(
                &vec![Tick {
                    index: 1,
                    liquidity_delta: 1,
                }],
                0
            ),
            None
        );
    }

    #[test]
    fn test_find_nearest_initialized_tick_index_many_ticks() {
        let sorted_ticks = vec![
            Tick {
                index: -100,
                liquidity_delta: 0,
            },
            Tick {
                index: -5,
                liquidity_delta: 0,
            },
            Tick {
                index: -4,
                liquidity_delta: 0,
            },
            Tick {
                index: 18,
                liquidity_delta: 0,
            },
            Tick {
                index: 23,
                liquidity_delta: 0,
            },
            Tick {
                index: 50,
                liquidity_delta: 0,
            },
        ];

        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -101),
            None
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -100),
            Some(0)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -99),
            Some(0)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -6),
            Some(0)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -5),
            Some(1)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -4),
            Some(2)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, -3),
            Some(2)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 0),
            Some(2)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 17),
            Some(2)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 18),
            Some(3)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 19),
            Some(3)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 22),
            Some(3)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 23),
            Some(4)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 24),
            Some(4)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 49),
            Some(4)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 50),
            Some(5)
        );
        assert_eq!(
            find_nearest_initialized_tick_index(&sorted_ticks, 51),
            Some(5)
        );
    }

    #[test]
    fn test_approximate_number_of_tick_spacings_crossed_for_doubling() {
        // 2x sqrt ratio increase ~= 4x price increase
        assert_eq!(
            approximate_number_of_tick_spacings_crossed(U256([0, 0, 1, 0]), U256([0, 0, 2, 0]), 1),
            1386295
        );
        assert_eq!(
            approximate_number_of_tick_spacings_crossed(MIN_SQRT_RATIO, MIN_SQRT_RATIO * 2, 1),
            1386295
        );
        assert_eq!(
            approximate_number_of_tick_spacings_crossed(MAX_SQRT_RATIO, MAX_SQRT_RATIO / 2, 1),
            1386295
        );
    }

    #[test]
    fn test_approximate_number_of_tick_spacings_crossed_for_doubling_big_tick_spacing() {
        // 2x sqrt ratio increase ~= 4x price increase
        assert_eq!(
            approximate_number_of_tick_spacings_crossed(U256([0, 0, 1, 0]), U256([0, 0, 2, 0]), 5),
            277259
        );
        assert_eq!(
            approximate_number_of_tick_spacings_crossed(MIN_SQRT_RATIO, MIN_SQRT_RATIO * 2, 3),
            462098
        );
        assert_eq!(
            approximate_number_of_tick_spacings_crossed(MAX_SQRT_RATIO, MAX_SQRT_RATIO / 2, 200),
            6931
        );
    }

    #[test]
    fn test_u256_to_fraction() {
        assert_eq!(
            u256_to_float_base_x128(U256([16403144882676588163, 1525053501570699700, 35, 0])),
            35.08267331597798
        );
        assert_eq!(
            u256_to_float_base_x128(U256([123456, 0, 0, 0])),
            3.628045764377908e-34
        );
        assert_eq!(
            u256_to_float_base_x128(U256([0, 123456, 0, 0])),
            6.692563170318522e-15
        );
        assert_eq!(u256_to_float_base_x128(U256([0, 0, 123456, 0])), 123456.0);
        assert_eq!(
            u256_to_float_base_x128(U256([0, 0, 0, 123456])),
            2.2773612363638864e24
        );
    }
}
