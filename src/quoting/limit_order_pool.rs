use crate::math::swap::is_price_increasing;
use crate::math::uint::U256;
use crate::quoting::base_pool::{BasePool, BasePoolQuoteError, BasePoolResources, BasePoolState};
use crate::quoting::types::{NodeKey, Pool, Quote, QuoteParams, Tick};
use crate::quoting::util::find_nearest_initialized_tick_index;
use alloc::vec::Vec;
use core::ops::Add;

#[derive(Clone, Copy)]
pub struct LimitOrderPoolState {
    pub base_pool_state: BasePoolState,
    // the maximum active tick index we've reached after a swap of token1 for token0
    // if None, then we haven't seen any swaps from token1 to token0
    // if Some(None), then we have swapped through all the ticks greater than the current active tick index
    pub max_tick_index_after_swap: Option<Option<usize>>,
    // the minimum active tick index we've reached after a swap of token0 for token1
    // if None, then we haven't seen any swaps from token0 to token1
    // if Some(None), then we have swapped through all the ticks less than the current active tick index
    pub min_tick_index_after_swap: Option<Option<usize>>,
}

#[derive(Default, Clone, Copy)]
pub struct LimitOrderPoolResources {
    pub base_pool_resources: BasePoolResources,
}

impl Add for LimitOrderPoolResources {
    type Output = LimitOrderPoolResources;

    fn add(self, rhs: Self) -> Self::Output {
        LimitOrderPoolResources {
            base_pool_resources: self.base_pool_resources + rhs.base_pool_resources,
        }
    }
}

pub struct LimitOrderPool {
    base_pool: BasePool,
}

pub const LIMIT_ORDER_TICK_SPACING: u32 = 128;

impl LimitOrderPool {
    pub fn new(
        token0: U256,
        token1: U256,
        extension: U256,
        sqrt_ratio: U256,
        tick: i32,
        liquidity: u128,
        sorted_ticks: Vec<Tick>,
    ) -> Self {
        LimitOrderPool {
            base_pool: BasePool::new(
                NodeKey {
                    token0,
                    token1,
                    fee: 0,
                    tick_spacing: LIMIT_ORDER_TICK_SPACING,
                    extension,
                },
                BasePoolState {
                    sqrt_ratio,
                    liquidity,
                    active_tick_index: find_nearest_initialized_tick_index(&sorted_ticks, tick),
                },
                sorted_ticks,
            ),
        }
    }
}

impl Pool for LimitOrderPool {
    type Resources = LimitOrderPoolResources;
    type State = LimitOrderPoolState;
    type QuoteError = BasePoolQuoteError;
    type Meta = ();

    fn get_key(&self) -> &NodeKey {
        self.base_pool.get_key()
    }

    fn get_state(&self) -> Self::State {
        LimitOrderPoolState {
            base_pool_state: self.base_pool.get_state(),
            max_tick_index_after_swap: None,
            min_tick_index_after_swap: None,
        }
    }

    fn quote(
        &self,
        params: QuoteParams<Self::State, Self::Meta>,
    ) -> Result<Quote<Self::Resources, Self::State>, Self::QuoteError> {
        if let Some(state) = params.override_state {
            let increasing = is_price_increasing(
                params.token_amount.amount,
                params.token_amount.token == self.get_key().token1,
            );

            panic!("todo");
        } else {
            let result = self.base_pool.quote(QuoteParams {
                sqrt_ratio_limit: params.sqrt_ratio_limit,
                override_state: params.override_state.map(|s| s.base_pool_state),
                token_amount: params.token_amount,
                meta: (),
            })?;

            let (min_tick_index_after_swap, max_tick_index_after_swap) =
                if result.is_price_increasing {
                    (None, Some(result.state_after.active_tick_index))
                } else {
                    (Some(result.state_after.active_tick_index), None)
                };

            Ok(Quote {
                calculated_amount: result.calculated_amount,
                consumed_amount: result.consumed_amount,
                execution_resources: LimitOrderPoolResources {
                    base_pool_resources: result.execution_resources,
                },
                fees_paid: result.fees_paid,
                is_price_increasing: result.is_price_increasing,
                state_after: LimitOrderPoolState {
                    base_pool_state: result.state_after,
                    max_tick_index_after_swap,
                    min_tick_index_after_swap,
                },
            })
        }
    }

    fn has_liquidity(&self) -> bool {
        self.base_pool.has_liquidity()
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
    use crate::math::tick::to_sqrt_ratio;
    use crate::math::uint::U256;
    use crate::quoting::limit_order_pool::{LimitOrderPool, LIMIT_ORDER_TICK_SPACING};
    use crate::quoting::types::{Pool, QuoteParams, Tick, TokenAmount};
    use alloc::vec;
    use num_traits::ToPrimitive;

    #[test]
    fn test_swap_one_for_zero_partial() {
        let liquidity: i128 = 10000000;
        let pool = LimitOrderPool::new(
            U256::from(1u32),
            U256::from(2u32),
            U256::from(3u32),
            to_sqrt_ratio(0).unwrap(),
            0,
            liquidity.unsigned_abs(),
            vec![
                Tick {
                    index: 0,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING.to_i32().unwrap(),
                    liquidity_delta: -liquidity,
                },
            ],
        );

        let quote = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: None,
                override_state: None,
                meta: (),
                token_amount: TokenAmount {
                    token: U256::from(2),
                    amount: 10000,
                },
            })
            .expect("Quote failed");

        assert_eq!(quote.fees_paid, 0);
        assert_eq!(quote.state_after.min_tick_index_after_swap, None);
        assert_eq!(quote.state_after.max_tick_index_after_swap, Some(None));
        assert_eq!(quote.consumed_amount, 641);
        assert_eq!(quote.calculated_amount, 639);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            1
        );
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .no_override_price_change,
            1
        );
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .tick_spacings_crossed,
            693147
        );
    }

    #[test]
    fn test_swap_one_for_zero_cross_multiple() {
        let liquidity: i128 = 10000000;
        let pool = LimitOrderPool::new(
            U256::from(1u32),
            U256::from(2u32),
            U256::from(3u32),
            to_sqrt_ratio(0).unwrap(),
            0,
            liquidity.unsigned_abs(),
            vec![
                Tick {
                    index: 0,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING.to_i32().unwrap(),
                    liquidity_delta: -liquidity,
                },
                Tick {
                    index: (LIMIT_ORDER_TICK_SPACING.to_i32().unwrap() * 2),
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: (LIMIT_ORDER_TICK_SPACING.to_i32().unwrap() * 3)
                        .to_i32()
                        .unwrap(),
                    liquidity_delta: -liquidity,
                },
            ],
        );

        let quote = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: None,
                override_state: None,
                meta: (),
                token_amount: TokenAmount {
                    token: U256::from(2),
                    amount: 1000,
                },
            })
            .expect("Quote failed");

        assert_eq!(quote.fees_paid, 0);
        assert_eq!(quote.state_after.min_tick_index_after_swap, None);
        assert_eq!(quote.state_after.max_tick_index_after_swap, Some(Some(2)));
        assert_eq!(quote.consumed_amount, 1000);
        assert_eq!(quote.calculated_amount, 997);
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .initialized_ticks_crossed,
            2
        );
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .no_override_price_change,
            1
        );
        assert_eq!(
            quote
                .execution_resources
                .base_pool_resources
                .tick_spacings_crossed,
            2
        );
    }
}
