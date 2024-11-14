use crate::math::uint::U256;
use crate::quoting::base_pool::{BasePool, BasePoolQuoteError, BasePoolResources, BasePoolState};
use crate::quoting::types::{NodeKey, Pool, Quote, QuoteParams, Tick};
use crate::quoting::util::find_nearest_initialized_tick_index;
use alloc::vec::Vec;
use core::ops::Add;
use num_traits::{ToPrimitive, Zero};

#[derive(Clone, Copy)]
pub struct LimitOrderPoolState {
    pub base_pool_state: BasePoolState,
    // the maximum tick that we have fully crossed
    pub max_tick_index_after_swap: Option<usize>,
    // the minimum tick that we have fully crossed
    pub min_tick_index_after_swap: Option<usize>,
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
        let (min_tick_after_swap, max_tick_after_swap) = params
            .override_state
            .map(|s| (s.min_tick_index_after_swap, s.max_tick_index_after_swap))
            .unwrap_or((None, None));

        let result = self.base_pool.quote(QuoteParams {
            sqrt_ratio_limit: params.sqrt_ratio_limit,
            override_state: params.override_state.map(|s| s.base_pool_state),
            token_amount: params.token_amount,
            meta: (),
        })?;

        let ix = result.state_after.active_tick_index;

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
                max_tick_index_after_swap: None,
                min_tick_index_after_swap: None,
            },
        })
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
mod tests {}
