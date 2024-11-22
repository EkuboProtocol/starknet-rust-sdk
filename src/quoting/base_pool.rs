use crate::math::swap::{compute_step, is_price_increasing, ComputeStepError};
use crate::math::tick::{to_sqrt_ratio, MAX_SQRT_RATIO, MIN_SQRT_RATIO};
use crate::math::uint::U256;
use crate::quoting::types::{NodeKey, Pool, Quote, QuoteParams, Tick};
use crate::quoting::util::approximate_number_of_tick_spacings_crossed;
use alloc::vec::Vec;
use core::ops::{Add, AddAssign};
use num_traits::Zero;

// Resources consumed during any swap execution.
#[derive(Clone, Copy, Default)]
pub struct BasePoolResources {
    pub no_override_price_change: u32,
    pub initialized_ticks_crossed: u32,
    pub tick_spacings_crossed: u32,
}

impl AddAssign for BasePoolResources {
    fn add_assign(&mut self, rhs: Self) {
        self.no_override_price_change += rhs.no_override_price_change;
        self.initialized_ticks_crossed += rhs.initialized_ticks_crossed;
        self.tick_spacings_crossed += rhs.tick_spacings_crossed;
    }
}

impl Add for BasePoolResources {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

pub const MAX_TICK_SPACING: u32 = 354892;
pub const MIN_TICK_AT_MAX_TICK_SPACING: i32 = -88368108;
pub const MAX_TICK_AT_MAX_TICK_SPACING: i32 = 88368108;
pub const MIN_SQRT_RATIO_AT_MAX_TICK_SPACING: U256 = U256([3580400339970425059, 1, 0, 0]);
pub const MAX_SQRT_RATIO_AT_MAX_TICK_SPACING: U256 = U256([
    6538062197914670769,
    14200015713685041661,
    15448319606494512814,
    0,
]);

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BasePoolQuoteError {
    InvalidToken,
    InvalidSqrtRatioLimit,
    InvalidTick(i32),
    FailedComputeSwapStep(ComputeStepError),
}

#[derive(Clone, Copy)]
pub struct BasePoolState {
    pub sqrt_ratio: U256,
    pub liquidity: u128,
    pub active_tick_index: Option<usize>,
}

pub struct BasePool {
    key: NodeKey,
    state: BasePoolState,
    sorted_ticks: Vec<Tick>,
}

impl BasePool {
    pub fn new(key: NodeKey, state: BasePoolState, sorted_ticks: Vec<Tick>) -> Self {
        assert!(key.token0 < key.token1, "token0 must be less than token1");
        assert!(!key.token0.is_zero(), "token0 must be non-zero");
        assert!(
            key.tick_spacing > 0,
            "tick spacing must be greater than zero"
        );
        assert!(
            key.tick_spacing <= MAX_TICK_SPACING,
            "tick spacing must be less than max tick spacing"
        );

        // check ticks are sorted in linear time
        let mut last_tick: Option<i32> = None;
        let mut total_liquidity: u128 = 0;
        let mut active_liquidity: u128 = 0;
        let spacing_i32 = key.tick_spacing as i32;
        for (i, tick) in sorted_ticks.iter().enumerate() {
            if let Some(last) = last_tick {
                assert!(tick.index > last, "ticks must be sorted");
            };
            assert!(
                (tick.index % spacing_i32).is_zero(),
                "all ticks must be multiple of tick_spacing"
            );
            last_tick = Some(tick.index);
            total_liquidity = if tick.liquidity_delta < 0 {
                total_liquidity - tick.liquidity_delta.unsigned_abs()
            } else {
                total_liquidity + tick.liquidity_delta.unsigned_abs()
            };

            if let Some(active_index) = state.active_tick_index {
                if i <= active_index {
                    if tick.liquidity_delta > 0 {
                        active_liquidity += tick.liquidity_delta.unsigned_abs();
                    } else {
                        active_liquidity -= tick.liquidity_delta.unsigned_abs();
                    }
                }
            }
        }
        assert!(total_liquidity.is_zero(), "total liquidity must be zero");
        assert_eq!(active_liquidity, state.liquidity, "active liquidity does not equal sum of liquidity deltas before or equal to active tick");

        if let Some(active) = state.active_tick_index {
            let tick = sorted_ticks
                .get(active)
                .expect("active tick index is out of bounds");

            assert!(
                to_sqrt_ratio(tick.index).expect("invalid active tick") <= state.sqrt_ratio,
                "sqrt_ratio of active tick is not less than or equal to current sqrt_ratio"
            );
        } else {
            if let Some(first) = sorted_ticks.first() {
                assert!(
                    state.sqrt_ratio
                        <= to_sqrt_ratio(first.index).expect("first tick has invalid index"),
                    "current sqrt_ratio must be lower than equal sqrt_ratio of first tick if active_tick_index is none"
                );
            }
        }

        Self {
            key,
            state,
            sorted_ticks,
        }
    }

    pub fn get_sorted_ticks(&self) -> &Vec<Tick> {
        &self.sorted_ticks
    }
}

impl Pool for BasePool {
    type Resources = BasePoolResources;
    type State = BasePoolState;
    type QuoteError = BasePoolQuoteError;
    type Meta = ();

    fn get_key(&self) -> &NodeKey {
        &self.key
    }

    fn get_state(&self) -> Self::State {
        self.state
    }

    fn quote(
        &self,
        params: QuoteParams<Self::State, Self::Meta>,
    ) -> Result<Quote<Self::Resources, Self::State>, Self::QuoteError> {
        let amount = params.token_amount.amount;
        let token = params.token_amount.token;
        let is_token1 = token == self.key.token1;

        if !is_token1 && token != self.key.token0 {
            return Err(BasePoolQuoteError::InvalidToken);
        }

        let state = if let Some(override_state) = params.override_state {
            override_state
        } else {
            self.state.clone()
        };

        if amount == 0 {
            return Ok(Quote {
                is_price_increasing: is_token1,
                consumed_amount: 0,
                calculated_amount: 0,
                execution_resources: Default::default(),
                state_after: state,
                fees_paid: 0,
            });
        }

        let is_increasing = is_price_increasing(amount, is_token1);

        let mut sqrt_ratio = state.sqrt_ratio;
        let mut liquidity = state.liquidity;
        let mut active_tick_index = state.active_tick_index;

        let sqrt_ratio_limit = if let Some(limit) = params.sqrt_ratio_limit {
            if is_increasing && limit < sqrt_ratio {
                return Err(BasePoolQuoteError::InvalidSqrtRatioLimit);
            }
            if !is_increasing && limit > sqrt_ratio {
                return Err(BasePoolQuoteError::InvalidSqrtRatioLimit);
            }
            if limit < MIN_SQRT_RATIO {
                return Err(BasePoolQuoteError::InvalidSqrtRatioLimit);
            }
            if limit > MAX_SQRT_RATIO {
                return Err(BasePoolQuoteError::InvalidSqrtRatioLimit);
            }
            limit
        } else {
            if is_increasing {
                MAX_SQRT_RATIO
            } else {
                MIN_SQRT_RATIO
            }
        };

        let mut calculated_amount: i128 = 0;
        let mut fees_paid: u128 = 0;
        let mut initialized_ticks_crossed: u32 = 0;
        let mut amount_remaining = amount;

        let starting_sqrt_ratio = sqrt_ratio;

        while amount_remaining != 0 && sqrt_ratio != sqrt_ratio_limit {
            let next_initialized_tick: Option<(usize, &Tick, U256)> = if is_increasing {
                if let Some(index) = active_tick_index {
                    if let Some(next) = self.sorted_ticks.get(index + 1) {
                        Some((
                            index + 1,
                            next,
                            to_sqrt_ratio(next.index)
                                .ok_or(BasePoolQuoteError::InvalidTick(next.index))?,
                        ))
                    } else {
                        None
                    }
                } else {
                    if let Some(next) = self.sorted_ticks.first() {
                        Some((
                            0,
                            next,
                            to_sqrt_ratio(next.index)
                                .ok_or(BasePoolQuoteError::InvalidTick(next.index))?,
                        ))
                    } else {
                        None
                    }
                }
            } else {
                if let Some(index) = active_tick_index {
                    if let Some(tick) = self.sorted_ticks.get(index) {
                        Some((
                            index,
                            tick,
                            to_sqrt_ratio(tick.index)
                                .ok_or(BasePoolQuoteError::InvalidTick(tick.index))?,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            let step_sqrt_ratio_limit =
                next_initialized_tick.map_or(sqrt_ratio_limit, |(_, _, next_ratio)| {
                    if (next_ratio < sqrt_ratio_limit) == is_increasing {
                        next_ratio
                    } else {
                        sqrt_ratio_limit
                    }
                });

            let step = compute_step(
                sqrt_ratio,
                liquidity,
                step_sqrt_ratio_limit,
                amount_remaining,
                is_token1,
                self.key.fee,
            )
            .map_err(BasePoolQuoteError::FailedComputeSwapStep)?;

            amount_remaining -= step.consumed_amount;
            calculated_amount += step.calculated_amount;
            fees_paid += step.fee_amount;
            sqrt_ratio = step.sqrt_ratio_next;

            if let Some((index, next_tick, tick_sqrt_ratio)) = next_initialized_tick {
                if sqrt_ratio == tick_sqrt_ratio {
                    active_tick_index = if is_increasing {
                        Some(index)
                    } else if !index.is_zero() {
                        Some(index - 1)
                    } else {
                        None
                    };

                    initialized_ticks_crossed += 1;

                    if (next_tick.liquidity_delta.signum() == 1) == is_increasing {
                        liquidity = liquidity + next_tick.liquidity_delta.unsigned_abs();
                    } else {
                        liquidity = liquidity - next_tick.liquidity_delta.unsigned_abs();
                    };
                }
            } else {
                active_tick_index = if is_increasing {
                    self.sorted_ticks.len().checked_sub(1)
                } else {
                    None
                };
            }
        }

        let resources = BasePoolResources {
            // we ignore changes from the override price because we assume the price has already changed
            no_override_price_change: if starting_sqrt_ratio == self.state.sqrt_ratio
                && starting_sqrt_ratio != sqrt_ratio
            {
                1
            } else {
                0
            },
            initialized_ticks_crossed,
            tick_spacings_crossed: approximate_number_of_tick_spacings_crossed(
                starting_sqrt_ratio,
                sqrt_ratio,
                self.key.tick_spacing,
            ),
        };

        let state_after = BasePoolState {
            sqrt_ratio,
            liquidity,
            active_tick_index,
        };

        Ok(Quote {
            is_price_increasing: is_increasing,
            consumed_amount: amount - amount_remaining,
            calculated_amount,
            execution_resources: resources,
            state_after,
            fees_paid,
        })
    }

    // Checks if the pool has any liquidity.
    fn has_liquidity(&self) -> bool {
        self.state.liquidity > 0 || !self.sorted_ticks.is_empty()
    }

    fn max_tick_with_liquidity(&self) -> Option<i32> {
        self.sorted_ticks.last().map(|t| t.index)
    }

    fn min_tick_with_liquidity(&self) -> Option<i32> {
        self.sorted_ticks.first().map(|t| t.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quoting::types::TokenAmount;
    use alloc::vec;

    const TOKEN0: U256 = U256([1, 0, 0, 0]);
    const TOKEN1: U256 = U256([2, 0, 0, 0]);

    fn node_key(tick_spacing: u32, fee: u128) -> NodeKey {
        NodeKey {
            token0: TOKEN0,
            token1: TOKEN1,
            tick_spacing,
            fee,
            extension: U256::zero(),
        }
    }

    mod constructor_validation {
        use super::{to_sqrt_ratio, vec, BasePool, BasePoolState, NodeKey, MAX_TICK_SPACING, U256};
        use crate::quoting::base_pool::MAX_TICK_AT_MAX_TICK_SPACING;
        use crate::quoting::types::Tick;

        #[test]
        #[should_panic(expected = "token0 must be less than token1")]
        fn test_token0_lt_token1() {
            BasePool::new(
                NodeKey {
                    token0: U256::zero(),
                    token1: U256::zero(),
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: 0,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: None,
                    liquidity: 0,
                },
                vec![],
            );
        }

        #[test]
        #[should_panic(expected = "token0 must be non-zero")]
        fn test_token0_non_zero() {
            BasePool::new(
                NodeKey {
                    token0: U256::zero(),
                    token1: U256::one(),
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: 0,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: None,
                    liquidity: 0,
                },
                vec![],
            );
        }

        #[test]
        #[should_panic(expected = "tick spacing must be greater than zero")]
        fn test_tick_spacing_non_zero() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: 0,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: None,
                    liquidity: 0,
                },
                vec![],
            );
        }

        #[test]
        #[should_panic(expected = "tick spacing must be less than max tick spacing")]
        fn test_tick_spacing_lte_max() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING + 1,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: None,
                    liquidity: 0,
                },
                vec![],
            );
        }

        #[test]
        #[should_panic(expected = "active tick index is out of bounds")]
        fn test_active_tick_index_within_range() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: Some(0),
                    liquidity: 0,
                },
                vec![],
            );
        }

        #[test]
        #[should_panic(expected = "ticks must be sorted")]
        fn test_ticks_must_be_sorted() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: Some(0),
                    liquidity: 1,
                },
                vec![
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: 0,
                    },
                    Tick {
                        index: 0,
                        liquidity_delta: 0,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(expected = "all ticks must be multiple of tick_spacing")]
        fn test_ticks_must_be_multiple_of_tick_spacing() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: Some(0),
                    liquidity: 1,
                },
                vec![
                    Tick {
                        index: -1,
                        liquidity_delta: 1,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING - 1,
                        liquidity_delta: -1,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(expected = "total liquidity must be zero")]
        fn test_ticks_must_total_to_zero_liquidity() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: Some(0),
                    liquidity: 2,
                },
                vec![
                    Tick {
                        index: 0,
                        liquidity_delta: 2,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: -1,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(expected = "active tick index is out of bounds")]
        fn test_active_tick_index_must_be_within_bounds() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: Some(2),
                    liquidity: 0,
                },
                vec![
                    Tick {
                        index: 0,
                        liquidity_delta: 2,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: -2,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(
            expected = "active liquidity does not equal sum of liquidity deltas before or equal to active tick"
        )]
        fn test_liquidity_equal_sum_of_deltas_active_ticks() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap(),
                    active_tick_index: Some(0),
                    liquidity: 0,
                },
                vec![
                    Tick {
                        index: 0,
                        liquidity_delta: 2,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: -2,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(
            expected = "sqrt_ratio of active tick is not less than or equal to current sqrt_ratio"
        )]
        fn test_active_tick_sqrt_ratio_is_lte_current_sqrt_ratio() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap() - 1,
                    active_tick_index: Some(0),
                    liquidity: 2,
                },
                vec![
                    Tick {
                        index: 0,
                        liquidity_delta: 2,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: -2,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(
            expected = "current sqrt_ratio must be lower than equal sqrt_ratio of first tick if active_tick_index is none"
        )]
        fn test_if_no_active_tick_sqrt_ratio_lte_first() {
            BasePool::new(
                NodeKey {
                    token0: U256::one(),
                    token1: U256::one() + 1,
                    extension: U256::zero(),
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                },
                BasePoolState {
                    sqrt_ratio: to_sqrt_ratio(0).unwrap() + 1,
                    active_tick_index: None,
                    liquidity: 0,
                },
                vec![
                    Tick {
                        index: 0,
                        liquidity_delta: 2,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: -2,
                    },
                ],
            );
        }
    }

    #[test]
    fn test_quote_zero_liquidity_token1_input() {
        let pool = BasePool::new(
            node_key(1, 0),
            BasePoolState {
                sqrt_ratio: U256([0, 0, 1, 0]),
                liquidity: 0u128,
                active_tick_index: None,
            },
            vec![],
        );

        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        };

        let quote = pool.quote(params).expect("Failed to get quote");

        assert_eq!(quote.calculated_amount, 0);
        assert_eq!(quote.execution_resources.initialized_ticks_crossed, 0);
    }

    #[test]
    fn test_quote_zero_liquidity_token0_input() {
        let pool = BasePool::new(
            node_key(1, 0),
            BasePoolState {
                sqrt_ratio: U256([0, 0, 1, 0]),
                liquidity: 0u128,
                active_tick_index: None,
            },
            vec![],
        );

        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        };

        let quote = pool.quote(params).expect("Failed to get quote");

        assert_eq!(quote.calculated_amount, 0);
        assert_eq!(quote.execution_resources.initialized_ticks_crossed, 0);
    }

    #[test]
    fn test_quote_liquidity_token1_input() {
        let sorted_ticks = vec![
            Tick {
                index: 0,
                liquidity_delta: 1_000_000_000,
            },
            Tick {
                index: 1,
                liquidity_delta: -1_000_000_000,
            },
        ];

        let pool = BasePool::new(
            node_key(1, 0),
            BasePoolState {
                sqrt_ratio: U256([0, 0, 1, 0]),
                liquidity: 1_000_000_000u128,
                active_tick_index: Some(0),
            },
            sorted_ticks,
        );

        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        };

        let quote = pool.quote(params).expect("Failed to get quote");

        assert_eq!(quote.calculated_amount, 499);
        assert_eq!(quote.execution_resources.initialized_ticks_crossed, 1);
    }

    #[test]
    fn test_quote_liquidity_token0_input() {
        let sorted_ticks = vec![
            Tick {
                index: 0,
                liquidity_delta: 1_000_000_000,
            },
            Tick {
                index: 1,
                liquidity_delta: -1_000_000_000,
            },
        ];

        let pool = BasePool::new(
            node_key(1, 0),
            BasePoolState {
                sqrt_ratio: to_sqrt_ratio(1).expect("Invalid tick"),
                liquidity: 0,
                active_tick_index: Some(1),
            },
            sorted_ticks,
        );

        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        };

        let quote = pool.quote(params).expect("Failed to get quote");

        assert_eq!(quote.calculated_amount, 499);
        assert_eq!(quote.execution_resources.initialized_ticks_crossed, 2);
    }

    #[test]
    fn test_example_failing_quote() {
        let pool = BasePool::new(
            node_key(100, 17014118346046923988514818429550592),
            BasePoolState {
                sqrt_ratio: U256([16035209758820767612, 757181812420893, 0, 0]),
                liquidity: 99999,
                active_tick_index: Some(16),
            },
            vec![
                Tick {
                    index: -88722000,
                    liquidity_delta: 99999,
                },
                Tick {
                    index: -24124600,
                    liquidity_delta: 103926982998885,
                },
                Tick {
                    index: -24124500,
                    liquidity_delta: -103926982998885,
                },
                Tick {
                    index: -20236100,
                    liquidity_delta: 20192651866847,
                },
                Tick {
                    index: -20235900,
                    liquidity_delta: 676843433645,
                },
                Tick {
                    index: -20235400,
                    liquidity_delta: 620315686813,
                },
                Tick {
                    index: -20235000,
                    liquidity_delta: 3899271022058,
                },
                Tick {
                    index: -20234900,
                    liquidity_delta: 1985516133391,
                },
                Tick {
                    index: -20233000,
                    liquidity_delta: 2459469409600,
                },
                Tick {
                    index: -20232100,
                    liquidity_delta: -20192651866847,
                },
                Tick {
                    index: -20231900,
                    liquidity_delta: -663892969024,
                },
                Tick {
                    index: -20231400,
                    liquidity_delta: -620315686813,
                },
                Tick {
                    index: -20231000,
                    liquidity_delta: -3516445235227,
                },
                Tick {
                    index: -20230900,
                    liquidity_delta: -1985516133391,
                },
                Tick {
                    index: -20229000,
                    liquidity_delta: -2459469409600,
                },
                Tick {
                    index: -20227900,
                    liquidity_delta: -12950464621,
                },
                Tick {
                    index: -20227000,
                    liquidity_delta: -382825786831,
                },
                Tick {
                    index: -2000,
                    liquidity_delta: 140308196,
                },
                Tick {
                    index: 2000,
                    liquidity_delta: -140308196,
                },
                Tick {
                    index: 88722000,
                    liquidity_delta: -99999,
                },
            ],
        );

        let quote = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000000,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: (),
            })
            .expect("Failed to get quote of token0");

        assert_eq!(quote.calculated_amount, 0);
        assert_eq!(quote.execution_resources.initialized_ticks_crossed, 0);

        let quote = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000000,
                    token: TOKEN1,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: (),
            })
            .expect("Failed to get quote of token1");

        assert_eq!(quote.consumed_amount, 1_000_000);
        assert_eq!(quote.calculated_amount, 2436479431);
        assert_eq!(quote.execution_resources.initialized_ticks_crossed, 2);
    }
}
