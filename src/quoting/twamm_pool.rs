use crate::math::twamm::sqrt_ratio::calculate_next_sqrt_ratio;
use crate::math::uint::U256;
use crate::quoting::base_pool::{
    BasePool, BasePoolQuoteError, BasePoolResources, BasePoolState,
    MAX_SQRT_RATIO_AT_MAX_TICK_SPACING, MAX_TICK_AT_MAX_TICK_SPACING, MAX_TICK_SPACING,
    MIN_SQRT_RATIO_AT_MAX_TICK_SPACING, MIN_TICK_AT_MAX_TICK_SPACING,
};
use crate::quoting::types::BlockTimestamp;
use crate::quoting::types::{NodeKey, Pool, Quote, QuoteParams, Tick, TokenAmount};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Add;
use num_traits::{ToPrimitive, Zero};

#[derive(Clone, Copy)]
pub struct TwammPoolState {
    pub base_pool_state: BasePoolState,
    pub token0_sale_rate: u128,
    pub token1_sale_rate: u128,
    pub last_execution_time: u64,
}

#[derive(Clone, Copy, Default)]
pub struct TwammPoolResources {
    pub base_pool_resources: BasePoolResources,
    // The number of seconds that passed since the last virtual order execution
    pub virtual_order_seconds_executed: u32,
    // The amount of order updates that were applied to the sale rate
    pub virtual_order_delta_times_crossed: u32,
    // Whether the virtual orders were executed or not (for a single swap, 1 or 0)
    pub virtual_orders_executed: u32,
}

impl Add for TwammPoolResources {
    type Output = TwammPoolResources;

    fn add(self, rhs: Self) -> Self::Output {
        TwammPoolResources {
            base_pool_resources: self.base_pool_resources + rhs.base_pool_resources,
            virtual_order_delta_times_crossed: self.virtual_order_delta_times_crossed
                + rhs.virtual_order_delta_times_crossed,
            virtual_order_seconds_executed: self.virtual_order_seconds_executed
                + rhs.virtual_order_seconds_executed,
            virtual_orders_executed: self.virtual_orders_executed + rhs.virtual_orders_executed,
        }
    }
}

#[derive(Clone)]
pub struct TwammSaleRateDelta {
    pub time: u64,
    pub sale_rate_delta0: i128,
    pub sale_rate_delta1: i128,
}

pub struct TwammPool {
    base_pool: BasePool,
    active_liquidity: u128,
    token0_sale_rate: u128,
    token1_sale_rate: u128,
    last_execution_time: u64,
    virtual_order_deltas: Vec<TwammSaleRateDelta>,
}

impl TwammPool {
    pub fn new(
        token0: U256,
        token1: U256,
        fee: u128,
        extension: U256,
        sqrt_ratio: U256,
        active_liquidity: u128,
        last_execution_time: u64,
        token0_sale_rate: u128,
        token1_sale_rate: u128,
        virtual_order_deltas: Vec<TwammSaleRateDelta>,
    ) -> Self {
        let signed_liquidity: i128 = active_liquidity.to_i128().expect("Liquidity overflow i128");

        let mut last_time = last_execution_time;
        let mut sr0: u128 = token0_sale_rate;
        let mut sr1: u128 = token1_sale_rate;
        for t in virtual_order_deltas.iter() {
            assert!(
                t.time > last_time,
                "Sale rate deltas are not ordered and greater than `last_execution_time`"
            );
            last_time = t.time;
            if t.sale_rate_delta0 < 0 {
                sr0 -= t.sale_rate_delta0.unsigned_abs();
            } else {
                sr0 += t.sale_rate_delta0.unsigned_abs();
            }
            if t.sale_rate_delta1 < 0 {
                sr1 -= t.sale_rate_delta1.unsigned_abs();
            } else {
                sr1 += t.sale_rate_delta1.unsigned_abs();
            }
        }

        assert!(
            sr0.is_zero() && sr1.is_zero(),
            "sum of current sale rate and sale rate deltas must be zero"
        );

        let (active_tick_index, sorted_ticks) = if active_liquidity.is_zero() {
            (None, vec![])
        } else {
            (
                Some(0),
                vec![
                    Tick {
                        index: MIN_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: signed_liquidity,
                    },
                    Tick {
                        index: MAX_TICK_AT_MAX_TICK_SPACING,
                        liquidity_delta: -signed_liquidity,
                    },
                ],
            )
        };

        TwammPool {
            active_liquidity,
            base_pool: BasePool::new(
                NodeKey {
                    token0,
                    token1,
                    fee,
                    tick_spacing: MAX_TICK_SPACING,
                    extension,
                },
                BasePoolState {
                    // we just force the pool state to always be within the bounds of min/max to simplify the state
                    // this does not change accuracy of quote results
                    // it just reduces accuracy of resource estimations in extreme cases by a negligible amount.
                    sqrt_ratio: sqrt_ratio
                        .min(MAX_SQRT_RATIO_AT_MAX_TICK_SPACING)
                        .max(MIN_SQRT_RATIO_AT_MAX_TICK_SPACING),
                    liquidity: active_liquidity,
                    active_tick_index,
                },
                sorted_ticks,
            ),
            virtual_order_deltas,
            last_execution_time,
            token0_sale_rate,
            token1_sale_rate,
        }
    }

    // Returns the list of sale rate deltas
    pub fn get_sale_rate_deltas(&self) -> &Vec<TwammSaleRateDelta> {
        &self.virtual_order_deltas
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TwammPoolQuoteError {
    ExecutionTimeExceedsBlockTime,
    FailedCalculateNextSqrtRatio,
    SaleAmountOverflow,
    TooMuchTimePassedSinceLastExecution,
    BasePoolQuoteError(BasePoolQuoteError),
}

impl Pool for TwammPool {
    type Resources = TwammPoolResources;
    type State = TwammPoolState;
    type QuoteError = TwammPoolQuoteError;
    type Meta = BlockTimestamp;

    fn get_key(&self) -> &NodeKey {
        self.base_pool.get_key()
    }

    fn get_state(&self) -> Self::State {
        TwammPoolState {
            base_pool_state: self.base_pool.get_state(),
            last_execution_time: self.last_execution_time,
            token0_sale_rate: self.token0_sale_rate,
            token1_sale_rate: self.token1_sale_rate,
        }
    }

    fn quote(
        &self,
        params: QuoteParams<Self::State, Self::Meta>,
    ) -> Result<Quote<Self::Resources, Self::State>, Self::QuoteError> {
        let QuoteParams {
            token_amount,
            sqrt_ratio_limit,
            override_state,
            meta,
        } = params;

        let current_time = meta;
        let initial_state = override_state.unwrap_or_else(|| self.get_state());

        let mut next_sqrt_ratio = initial_state.base_pool_state.sqrt_ratio;
        let mut token0_sale_rate = initial_state.token0_sale_rate;
        let mut token1_sale_rate = initial_state.token1_sale_rate;
        let mut last_execution_time = initial_state.last_execution_time;

        if current_time < last_execution_time {
            return Err(TwammPoolQuoteError::ExecutionTimeExceedsBlockTime);
        }

        let mut virtual_order_delta_times_crossed = 0;
        let mut next_sale_rate_delta_index = self
            .virtual_order_deltas
            .iter()
            .position(|srd| srd.time > last_execution_time)
            .unwrap_or(self.virtual_order_deltas.len());

        let mut base_pool_state_override = override_state.map(|s| s.base_pool_state);
        let mut base_pool_execution_resources: BasePoolResources = Default::default();

        let NodeKey {
            fee,
            token0,
            token1,
            tick_spacing: _,
            extension: _,
        } = self.base_pool.get_key();

        while last_execution_time != current_time {
            let sale_rate_delta = self.virtual_order_deltas.get(next_sale_rate_delta_index);

            let next_execution_time = sale_rate_delta
                .map(|srd| srd.time.min(current_time))
                .unwrap_or(current_time);

            let time_elapsed = next_execution_time - last_execution_time;
            if time_elapsed > u32::MAX.into() {
                return Err(TwammPoolQuoteError::TooMuchTimePassedSinceLastExecution);
            }

            // we know this will never overflow because token0_sale_rate is a u128 and time_elapsed is a u32
            let amount0: u128 =
                ((U256::from(token0_sale_rate) * U256::from(time_elapsed)) >> 32).low_u128();
            let amount1: u128 =
                ((U256::from(token1_sale_rate) * U256::from(time_elapsed)) >> 32).low_u128();

            if amount0 > 0 && amount1 > 0 {
                let current_sqrt_ratio = next_sqrt_ratio
                    .min(MAX_SQRT_RATIO_AT_MAX_TICK_SPACING)
                    .max(MIN_SQRT_RATIO_AT_MAX_TICK_SPACING);

                next_sqrt_ratio = calculate_next_sqrt_ratio(
                    current_sqrt_ratio,
                    // we explicitly do not use the liquidity state variable, we always calculate this with active liquidity
                    self.active_liquidity,
                    token0_sale_rate,
                    token1_sale_rate,
                    time_elapsed as u32,
                    *fee,
                )
                .ok_or(TwammPoolQuoteError::FailedCalculateNextSqrtRatio)?;

                let (token, amount) = if current_sqrt_ratio < next_sqrt_ratio {
                    (token1, amount1)
                } else {
                    (token0, amount0)
                };

                let quote = self
                    .base_pool
                    .quote(QuoteParams {
                        token_amount: TokenAmount {
                            amount: amount
                                .to_i128()
                                .ok_or(TwammPoolQuoteError::SaleAmountOverflow)?,
                            token: *token,
                        },
                        sqrt_ratio_limit: Some(next_sqrt_ratio),
                        override_state: base_pool_state_override,
                        meta: (),
                    })
                    .map_err(TwammPoolQuoteError::BasePoolQuoteError)?;

                base_pool_state_override = Some(quote.state_after);
                base_pool_execution_resources += quote.execution_resources;
            } else if amount0 > 0 || amount1 > 0 {
                let (amount, is_token1, sqrt_ratio_limit) = if amount0 != 0 {
                    (amount0, false, MIN_SQRT_RATIO_AT_MAX_TICK_SPACING)
                } else {
                    (amount1, true, MAX_SQRT_RATIO_AT_MAX_TICK_SPACING)
                };

                let token = if is_token1 {
                    self.base_pool.get_key().token1
                } else {
                    self.base_pool.get_key().token0
                };

                let quote = self
                    .base_pool
                    .quote(QuoteParams {
                        token_amount: TokenAmount {
                            amount: amount
                                .to_i128()
                                .ok_or(TwammPoolQuoteError::SaleAmountOverflow)?,
                            token,
                        },
                        sqrt_ratio_limit: Some(sqrt_ratio_limit),
                        override_state: base_pool_state_override,
                        meta: (),
                    })
                    .map_err(TwammPoolQuoteError::BasePoolQuoteError)?;

                base_pool_state_override = Some(quote.state_after);
                base_pool_execution_resources =
                    base_pool_execution_resources + quote.execution_resources;

                next_sqrt_ratio = quote.state_after.sqrt_ratio;
            }

            if let Some(next_delta) = sale_rate_delta {
                if next_delta.time == next_execution_time {
                    token0_sale_rate = if next_delta.sale_rate_delta0 < 0 {
                        token0_sale_rate - next_delta.sale_rate_delta0.unsigned_abs()
                    } else {
                        token0_sale_rate + next_delta.sale_rate_delta0.unsigned_abs()
                    };
                    token1_sale_rate = if next_delta.sale_rate_delta1 < 0 {
                        token1_sale_rate - next_delta.sale_rate_delta1.unsigned_abs()
                    } else {
                        token1_sale_rate + next_delta.sale_rate_delta1.unsigned_abs()
                    };
                    next_sale_rate_delta_index += 1;
                    virtual_order_delta_times_crossed += 1;
                }
            }

            last_execution_time = next_execution_time;
        }

        let final_quote = self
            .base_pool
            .quote(QuoteParams {
                token_amount,
                sqrt_ratio_limit,
                meta: (),
                override_state: base_pool_state_override,
            })
            .map_err(TwammPoolQuoteError::BasePoolQuoteError)?;

        Ok(Quote {
            is_price_increasing: final_quote.is_price_increasing,
            consumed_amount: final_quote.consumed_amount,
            calculated_amount: final_quote.calculated_amount,
            fees_paid: final_quote.fees_paid,
            execution_resources: TwammPoolResources {
                base_pool_resources: base_pool_execution_resources
                    + final_quote.execution_resources,
                virtual_order_seconds_executed: (current_time - initial_state.last_execution_time)
                    as u32,
                virtual_order_delta_times_crossed,
                virtual_orders_executed: if current_time > initial_state.last_execution_time {
                    1
                } else {
                    0
                },
            },
            state_after: TwammPoolState {
                base_pool_state: final_quote.state_after,
                token0_sale_rate,
                token1_sale_rate,
                last_execution_time: current_time,
            },
        })
    }

    fn has_liquidity(&self) -> bool {
        !self.active_liquidity.is_zero()
    }

    fn max_tick_with_liquidity(&self) -> Option<i32> {
        self.base_pool.max_tick_with_liquidity()
    }

    fn min_tick_with_liquidity(&self) -> Option<i32> {
        self.base_pool.min_tick_with_liquidity()
    }
}

#[cfg(test)]
mod tests {
    use crate::math::tick::{to_sqrt_ratio, MAX_SQRT_RATIO, MIN_SQRT_RATIO};
    use crate::math::uint::U256;
    use crate::quoting::base_pool::{
        MAX_SQRT_RATIO_AT_MAX_TICK_SPACING, MIN_SQRT_RATIO_AT_MAX_TICK_SPACING,
    };
    use crate::quoting::twamm_pool::{TwammPool, TwammSaleRateDelta};
    use crate::quoting::types::{Pool, QuoteParams, TokenAmount};
    use alloc::vec;

    const TOKEN0: U256 = U256([1, 0, 0, 0]);
    const TOKEN1: U256 = U256([2, 0, 0, 0]);

    mod constructor_validation {
        use crate::math::tick::{MAX_SQRT_RATIO, MIN_SQRT_RATIO};
        use crate::math::uint::U256;
        use crate::quoting::base_pool::{
            MAX_SQRT_RATIO_AT_MAX_TICK_SPACING, MIN_SQRT_RATIO_AT_MAX_TICK_SPACING,
        };
        use crate::quoting::twamm_pool::{TwammPool, TwammSaleRateDelta};
        use crate::quoting::types::Pool;
        use alloc::vec;

        #[test]
        fn test_max_price_constructor() {
            assert_eq!(
                TwammPool::new(
                    U256::one(),
                    U256::one() + 1,
                    0,
                    U256::zero(),
                    MAX_SQRT_RATIO,
                    1,
                    0,
                    0,
                    0,
                    vec![]
                )
                .get_state()
                .base_pool_state
                .liquidity,
                1
            );
        }

        #[test]
        fn test_min_price_constructor() {
            assert_eq!(
                TwammPool::new(
                    U256::one(),
                    U256::one() + 1,
                    0,
                    U256::zero(),
                    MIN_SQRT_RATIO,
                    1,
                    0,
                    0,
                    0,
                    vec![]
                )
                .get_state()
                .base_pool_state
                .liquidity,
                1
            );
        }

        #[test]
        fn test_min_sqrt_ratio_at_max_tick_spacing() {
            assert_eq!(
                TwammPool::new(
                    U256::one(),
                    U256::one() + 1,
                    0,
                    U256::zero(),
                    MIN_SQRT_RATIO_AT_MAX_TICK_SPACING,
                    1,
                    0,
                    0,
                    0,
                    vec![]
                )
                .get_state()
                .base_pool_state
                .liquidity,
                1
            );
        }

        #[test]
        fn test_max_sqrt_ratio_at_max_tick_spacing() {
            assert_eq!(
                TwammPool::new(
                    U256::one(),
                    U256::one() + 1,
                    0,
                    U256::zero(),
                    MAX_SQRT_RATIO_AT_MAX_TICK_SPACING,
                    1,
                    0,
                    0,
                    0,
                    vec![]
                )
                .get_state()
                .base_pool_state
                .liquidity,
                1
            );
        }

        #[test]
        #[should_panic(
            expected = "Sale rate deltas are not ordered and greater than `last_execution_time`"
        )]
        fn test_sale_rate_deltas_must_be_gt_last_execution_time() {
            TwammPool::new(
                U256::one(),
                U256::one() + 1,
                0,
                U256::zero(),
                MAX_SQRT_RATIO_AT_MAX_TICK_SPACING,
                1,
                0,
                0,
                0,
                vec![TwammSaleRateDelta {
                    time: 0,
                    sale_rate_delta0: 0,
                    sale_rate_delta1: 0,
                }],
            );
        }

        #[test]
        #[should_panic(
            expected = "Sale rate deltas are not ordered and greater than `last_execution_time`"
        )]
        fn test_sale_rate_deltas_must_be_ordered() {
            TwammPool::new(
                U256::one(),
                U256::one() + 1,
                0,
                U256::zero(),
                MAX_SQRT_RATIO_AT_MAX_TICK_SPACING,
                1,
                0,
                0,
                0,
                vec![
                    TwammSaleRateDelta {
                        time: 2,
                        sale_rate_delta0: 0,
                        sale_rate_delta1: 0,
                    },
                    TwammSaleRateDelta {
                        time: 1,
                        sale_rate_delta0: 0,
                        sale_rate_delta1: 0,
                    },
                ],
            );
        }

        #[test]
        #[should_panic(expected = "sum of current sale rate and sale rate deltas must be zero")]
        fn test_sale_rate_deltas_must_sum_to_zero() {
            TwammPool::new(
                U256::one(),
                U256::one() + 1,
                0,
                U256::zero(),
                MAX_SQRT_RATIO_AT_MAX_TICK_SPACING,
                1,
                0,
                54,
                2,
                vec![
                    TwammSaleRateDelta {
                        time: 1,
                        sale_rate_delta0: 0,
                        sale_rate_delta1: 1,
                    },
                    TwammSaleRateDelta {
                        time: 2,
                        sale_rate_delta0: 1,
                        sale_rate_delta1: 0,
                    },
                ],
            );
        }

        #[test]
        fn test_sale_rate_deltas_sum_to_zero() {
            TwammPool::new(
                U256::one(),
                U256::one() + 1,
                0,
                U256::zero(),
                MAX_SQRT_RATIO_AT_MAX_TICK_SPACING,
                1,
                0,
                23,
                35,
                vec![
                    TwammSaleRateDelta {
                        time: 1,
                        sale_rate_delta0: -23,
                        sale_rate_delta1: 0,
                    },
                    TwammSaleRateDelta {
                        time: 2,
                        sale_rate_delta0: 0,
                        sale_rate_delta1: -35,
                    },
                ],
            );
        }
    }

    #[test]
    fn zero_sale_rates_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0,
            U256::one(),
            to_sqrt_ratio(1).unwrap(),
            1_000_000_000,
            0,
            0,
            0,
            vec![],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000.into(),
                token: TOKEN0,
            },
            sqrt_ratio_limit: Some(MIN_SQRT_RATIO),
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 999);
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn zero_sale_rates_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0,
            U256::one(),
            to_sqrt_ratio(1).unwrap(),
            100_000,
            0,
            0,
            0,
            vec![],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000.into(),
                token: TOKEN1,
            },
            sqrt_ratio_limit: Some(MAX_SQRT_RATIO),
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 990);
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn non_zero_sale_rate_token0_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0,
            U256::one(),
            to_sqrt_ratio(1).unwrap(),
            1_000_000,
            0,
            0,
            1 << 32,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: 0,
                sale_rate_delta1: -(1 << 32),
            }],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000.into(),
                token: TOKEN0,
            },
            sqrt_ratio_limit: Some(MIN_SQRT_RATIO),
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 998);
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn non_zero_sale_rate_token1_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0,
            U256::one(),
            to_sqrt_ratio(1).unwrap(),
            1_000_000,
            0,
            1 << 32,
            0,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(1 << 32),
                sale_rate_delta1: 0,
            }],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000.into(),
                token: TOKEN1,
            },
            sqrt_ratio_limit: Some(MAX_SQRT_RATIO),
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(
            quote.calculated_amount, /*expected calculated amount*/
            999
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn non_zero_sale_rate_token0_max_price_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0,
            U256::one(),
            MAX_SQRT_RATIO,
            1_000_000,
            0,
            0,
            1 << 32,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: 0,
                sale_rate_delta1: -(1 << 32),
            }],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000.into(),
                token: TOKEN1,
            },
            sqrt_ratio_limit: Some(MAX_SQRT_RATIO),
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(
            quote.calculated_amount, /*expected calculated amount*/
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }
    #[test]
    fn zero_sale_rate_token0_close_to_max_usable_price_deltas_move_to_usable_price_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            MAX_SQRT_RATIO_AT_MAX_TICK_SPACING + 1,
            1_000_000u128,
            0,
            0,
            1 << 32,
            vec![
                TwammSaleRateDelta {
                    sale_rate_delta0: 100_000i128 * (1 << 32),
                    sale_rate_delta1: 0,
                    time: 16u64,
                },
                TwammSaleRateDelta {
                    time: u64::MAX,
                    sale_rate_delta0: -100_000 * (1 << 32),
                    sale_rate_delta1: -(1 << 32),
                },
            ],
        );

        let quote = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN1,
                },
                meta: 32,
                sqrt_ratio_limit: None,
                override_state: None,
            })
            .expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 2555);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            1
        );
    }

    #[test]
    fn zero_sale_rate_token1_close_to_min_usable_price_deltas_move_to_usable_price_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            MIN_SQRT_RATIO_AT_MAX_TICK_SPACING,
            1_000_000u128,
            0u64,
            1 << 32,
            0u128,
            vec![
                TwammSaleRateDelta {
                    sale_rate_delta0: 0i128,
                    sale_rate_delta1: 100_000 * (1 << 32),
                    time: 16u64,
                },
                TwammSaleRateDelta {
                    time: u64::MAX,
                    sale_rate_delta0: -(1 << 32),
                    sale_rate_delta1: -100_000 * (1 << 32),
                },
            ],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN1,
            },
            meta: 32,
            sqrt_ratio_limit: None,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 390);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            1
        );
    }

    #[test]
    fn zero_sale_rate_token0_close_to_max_usable_price_deltas_move_to_usable_price_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            MAX_SQRT_RATIO_AT_MAX_TICK_SPACING,
            1_000_000,
            0,
            0,
            1 << 32,
            vec![
                TwammSaleRateDelta {
                    sale_rate_delta0: 100_000 * (1 << 32),
                    sale_rate_delta1: 0,
                    time: 16,
                },
                TwammSaleRateDelta {
                    time: u64::MAX,
                    sale_rate_delta0: -100_000 * (1 << 32),
                    sale_rate_delta1: -(1 << 32),
                },
            ],
        );

        let quote = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN0,
                },
                meta: 32,
                sqrt_ratio_limit: None,
                override_state: None,
            })
            .expect("swap succeeds");

        assert_eq!(quote.calculated_amount, 390);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            1
        );
    }

    #[test]
    fn zero_sale_rate_token1_close_to_min_usable_price_deltas_move_to_usable_price_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            MIN_SQRT_RATIO_AT_MAX_TICK_SPACING,
            1_000_000,
            0,
            1 << 32,
            0,
            vec![
                TwammSaleRateDelta {
                    sale_rate_delta0: 0i128,
                    sale_rate_delta1: (100_000u128 * (1 << 32)) as i128,
                    time: 16u64,
                },
                TwammSaleRateDelta {
                    time: u64::MAX,
                    sale_rate_delta0: -(1 << 32),
                    sale_rate_delta1: -100_000 * (1 << 32),
                },
            ],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            meta: 32,
            sqrt_ratio_limit: None,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 2555);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            1
        );
    }

    #[test]
    fn one_e18_sale_rates_no_sale_rate_deltas_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            1 << 32u128,
            1 << 32u128,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(1 << 32),
                sale_rate_delta1: -(1 << 32),
            }], // No sale rate deltas
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 990);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn one_e18_sale_rates_no_sale_rate_deltas_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            1 << 32u128,
            1 << 32u128,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(1 << 32),
                sale_rate_delta1: -(1 << 32),
            }], // No sale rate deltas
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 989);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn token0_sale_rate_greater_than_token1_sale_rate_no_sale_rate_deltas_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            1_000u128,
            0u64,
            10 << 32u128,
            1 << 32u128,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(10 << 32),
                sale_rate_delta1: -(1 << 32),
            }], // No sale rate deltas
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 717);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn token1_sale_rate_greater_than_token0_sale_rate_no_sale_rate_deltas_quote_token1() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            1 << 32u128,
            10 << 32u128,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(1 << 32),
                sale_rate_delta1: -(10 << 32),
            }], // No sale rate deltas
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 984);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn token0_sale_rate_greater_than_token1_sale_rate_no_sale_rate_deltas_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            10 << 32u128,
            1 << 32u128,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(10 << 32),
                sale_rate_delta1: -(1 << 32),
            }], // No sale rate deltas
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 983);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn token1_sale_rate_greater_than_token0_sale_rate_no_sale_rate_deltas_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            1 << 32u128,
            10 << 32u128,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -(1 << 32),
                sale_rate_delta1: -(10 << 32),
            }], // No sale rate deltas
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 994);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn sale_rate_deltas_goes_to_zero_halfway_through_execution_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            1 << 32u128,
            1 << 32u128,
            vec![TwammSaleRateDelta {
                sale_rate_delta0: -(2u128.pow(32) as i128),
                sale_rate_delta1: -(2u128.pow(32) as i128),
                time: 16u64,
            }],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 989);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            1
        );
    }

    #[test]
    fn sale_rate_deltas_doubles_halfway_through_execution_quote_token0() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(1i32).unwrap(),
            100_000u128,
            0u64,
            1 << 32u128,
            1 << 32u128,
            vec![
                TwammSaleRateDelta {
                    sale_rate_delta0: 2i128.pow(32),
                    sale_rate_delta1: 2i128.pow(32),
                    time: 16u64,
                },
                TwammSaleRateDelta {
                    time: u64::MAX,
                    sale_rate_delta0: -(1 << 33),
                    sale_rate_delta1: -(1 << 33),
                },
            ],
        );

        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            meta: 32,
            override_state: None,
        });

        let quote = result.expect("Quote should succeed");

        assert_eq!(quote.calculated_amount, 989);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            0
        );
        assert_eq!(quote.execution_resources.virtual_order_seconds_executed, 32);
        assert_eq!(
            quote.execution_resources.virtual_order_delta_times_crossed,
            1
        );
    }

    #[test]
    fn price_after_no_swap() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(693147i32).unwrap(),
            70_710_696_755_630_728_101_718_334u128,
            0,
            10_526_880_627_450_980_392_156_862_745,
            10_526_880_627_450_980_392_156_862_745,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -10_526_880_627_450_980_392_156_862_745,
                sale_rate_delta1: -10_526_880_627_450_980_392_156_862_745,
            }],
        );

        // First quote: no swap
        let first = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 0,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: Some(to_sqrt_ratio(693147i32).unwrap()),
                meta: 43_200,
                override_state: None,
            })
            .expect("first swap succeeds");

        // Second quote: no swap after full day
        pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 0,
                token: TOKEN0,
            },
            sqrt_ratio_limit: Some(to_sqrt_ratio(693147).unwrap()),
            meta: 86_400,
            override_state: None,
        })
        .expect("second swap succeeds");

        // Third quote: using override_state from first quote
        pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 0,
                token: TOKEN0,
            },
            sqrt_ratio_limit: Some(to_sqrt_ratio(693147).unwrap()),
            meta: 86_400,
            override_state: Some(first.state_after),
        })
        .expect("third is ok");
    }

    #[test]
    fn moody_testing_examples() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(693147i32).unwrap(), // ~=2
            1_000_000_000_000_000_000_000u128, // 10^21
            60u64,
            10u128.pow(18) << 32,
            10u128.pow(18) << 32,
            vec![TwammSaleRateDelta {
                sale_rate_delta0: -((10u128.pow(18) << 32) as i128),
                sale_rate_delta1: -((10u128.pow(18) << 32) as i128),
                time: 120u64,
            }],
        );

        // Quote at time 60 (0 seconds pass)
        pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 0,
                token: TOKEN0,
            },
            meta: 60,
            sqrt_ratio_limit: None,
            override_state: None,
        })
        .expect("quote after 60 seconds");

        pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 0,
                token: TOKEN0,
            },
            meta: 90,
            sqrt_ratio_limit: None,
            override_state: None,
        })
        .expect("quote after 90 seconds");

        let fully_executed_twamm = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 0,
                    token: TOKEN0,
                },
                meta: 120,
                sqrt_ratio_limit: None,
                override_state: None,
            })
            .expect("quote after 120 seconds");

        let state_after_fully_executed = fully_executed_twamm.state_after;

        let quote_token0_with_override = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 10u128.pow(18) as i128,
                    token: TOKEN0,
                },
                meta: 120,
                sqrt_ratio_limit: None,
                override_state: Some(state_after_fully_executed),
            })
            .expect("quote with override");

        assert_eq!(
            quote_token0_with_override.calculated_amount,
            pool.base_pool
                .quote(QuoteParams {
                    token_amount: TokenAmount {
                        token: TOKEN0,
                        amount: 10u128.pow(18) as i128,
                    },
                    meta: (),
                    override_state: Some(state_after_fully_executed.base_pool_state),
                    sqrt_ratio_limit: None,
                })
                .expect("base pool quote")
                .calculated_amount
        );

        let quote_token1_with_override = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: 10u128.pow(18) as i128,
                    token: TOKEN1,
                },
                sqrt_ratio_limit: Some(to_sqrt_ratio(693147).unwrap()),
                meta: 120,
                override_state: Some(state_after_fully_executed),
            })
            .expect("quote token1 with override");

        // Replace with actual expected value comparison
        assert_eq!(
            quote_token1_with_override.calculated_amount,
            pool.base_pool
                .quote(QuoteParams {
                    token_amount: TokenAmount {
                        token: TOKEN1,
                        amount: 10u128.pow(18) as i128,
                    },
                    meta: (),
                    override_state: Some(fully_executed_twamm.state_after.base_pool_state),
                    sqrt_ratio_limit: Some(to_sqrt_ratio(693147).unwrap()),
                })
                .unwrap()
                .calculated_amount
        );
    }

    #[test]
    fn compare_to_contract_output() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(693147i32).unwrap(),
            70_710_696_755_630_728_101_718_334u128,
            0,
            10_526_880_627_450_980_392_156_862_745,
            10_526_880_627_450_980_392_156_862_745,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -10_526_880_627_450_980_392_156_862_745,
                sale_rate_delta1: -10_526_880_627_450_980_392_156_862_745,
            }],
        );

        // First swap
        let first_swap = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: (10_000u128 * 10u128.pow(18)) as i128,
                    token: TOKEN0,
                },
                meta: 2_040,
                sqrt_ratio_limit: None,
                override_state: None,
            })
            .expect("first swap succeeds");

        assert_eq!(first_swap.calculated_amount, 19993991114278719145825);
        assert_eq!(first_swap.consumed_amount, 10000000000000000000000);
        assert_eq!(
            first_swap
                .execution_resources
                .virtual_order_seconds_executed,
            2040
        );
        assert_eq!(
            first_swap
                .execution_resources
                .virtual_order_delta_times_crossed,
            0
        );

        // Second swap using override_state from first swap
        let second_swap = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: (10_000u128 * 10u128.pow(18)) as i128,
                    token: TOKEN0,
                },
                meta: 2_100,
                sqrt_ratio_limit: None,
                override_state: Some(first_swap.state_after),
            })
            .expect("second swap succeeds");

        assert_eq!(second_swap.calculated_amount, 19985938387207888688304);
        assert_eq!(second_swap.consumed_amount, 10000000000000000000000);
        assert_eq!(
            second_swap
                .execution_resources
                .virtual_order_seconds_executed,
            60
        );
        assert_eq!(
            second_swap
                .execution_resources
                .virtual_order_delta_times_crossed,
            0
        );
    }

    #[test]
    fn second_swap_in_opposite_direction() {
        let pool = TwammPool::new(
            TOKEN0,
            TOKEN1,
            0u128,
            U256::from(1u8),
            to_sqrt_ratio(693147).unwrap(),
            70_710_696_755_630_728_101_718_334u128,
            0,
            10_526_880_627_450_980_392_156_862_745,
            10_526_880_627_450_980_392_156_862_745,
            vec![TwammSaleRateDelta {
                time: u64::MAX,
                sale_rate_delta0: -10_526_880_627_450_980_392_156_862_745,
                sale_rate_delta1: -10_526_880_627_450_980_392_156_862_745,
            }],
        );

        // First swap
        let first_swap = pool
            .quote(QuoteParams {
                token_amount: TokenAmount {
                    amount: (10_000u128 * 10u128.pow(18)) as i128,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: None,
                meta: 2_040,
                override_state: None,
            })
            .expect("first swap succeeds");

        // Second swap in opposite direction using override_state from first swap
        pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: (10_000u128 * 10u128.pow(18)) as i128,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            meta: 2_100,
            override_state: Some(first_swap.state_after),
        })
        .expect("second swap succeeds");
    }
}
