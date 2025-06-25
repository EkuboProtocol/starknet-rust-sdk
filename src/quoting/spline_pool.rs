use crate::math::uint::U256;
use crate::quoting::base_pool::{
    BasePool, BasePoolResources, BasePoolState, BasePoolQuoteError
};
use crate::quoting::types::{NodeKey, Pool, Quote, QuoteParams, Tick};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Add;

#[derive(Clone, Copy)]
pub struct SplinePoolState {
    pub base_pool_state: BasePoolState,
    pub liquidity_factor: u128,
    pub total_shares: U256,
}

#[derive(Default, Clone, Copy)]
pub struct SplinePoolResources {
    pub base_pool_resources: BasePoolResources,
    pub fee_compounds: u32,
}

impl Add for SplinePoolResources {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            base_pool_resources: self.base_pool_resources + rhs.base_pool_resources,
            fee_compounds: self.fee_compounds + rhs.fee_compounds,
        }
    }
}

pub struct SplinePool {
    base_pool: BasePool,
    liquidity_factor: u128,
    total_shares: U256,
}

impl SplinePool {
    pub fn new(key: NodeKey, state: SplinePoolState, sorted_ticks: Vec<Tick>) -> Self {
        SplinePool {
            base_pool: BasePool::new(
                key,
                state.base_pool_state,
                sorted_ticks,
            ),
            liquidity_factor: state.liquidity_factor,
            total_shares: state.liquidity_factor.into(),
        }
    }
}

impl Pool for SplinePool {
    type Resources = SplinePoolResources;
    type State = SplinePoolState;
    type QuoteError = BasePoolQuoteError;
    type Meta = ();

    fn get_key(&self) -> &NodeKey {
        self.base_pool.get_key()
    }

    fn get_state(&self) -> Self::State {
        SplinePoolState {
            base_pool_state: self.base_pool.get_state(),
            liquidity_factor: self.liquidity_factor,
            total_shares: self.liquidity_factor.into(),
        }
    }

    fn quote(
        &self,
        params: QuoteParams<Self::State, Self::Meta>,
    ) -> Result<Quote<Self::Resources, Self::State>, Self::QuoteError> {
        let fee_compounds = 0;
        
        let liquidity_factor = params.override_state.map_or(self.liquidity_factor, |os| os.liquidity_factor);
        
        let result = self.base_pool.quote(QuoteParams {
            sqrt_ratio_limit: params.sqrt_ratio_limit,
            override_state: params.override_state.map(|s| s.base_pool_state),
            token_amount: params.token_amount,
            meta: (),
        })?;
        
        Ok(Quote {
            calculated_amount: result.calculated_amount,
            consumed_amount: result.consumed_amount,
            execution_resources: SplinePoolResources {
                base_pool_resources: result.execution_resources,
                fee_compounds,
            },
            fees_paid: result.fees_paid,
            is_price_increasing: result.is_price_increasing,
            state_after: SplinePoolState {
                base_pool_state: result.state_after,
                liquidity_factor,
                total_shares: self.total_shares,
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
mod tests {
    use super::*;
    use crate::quoting::base_pool::BasePoolState;
    use crate::quoting::types::TokenAmount;
    use crate::math::tick::{to_sqrt_ratio};
    use crate::math::uint::U256;

    const TOKEN0: U256 = U256([1, 0, 0, 0]);
    const TOKEN1: U256 = U256([2, 0, 0, 0]);
    const FEE_ONE_BPS: u128 = 34_028_236_692_093_846_346_337_460_743_176_821u128;


    fn cauchy_liquidity_at_tick(l0: u128, gamma: u128, mu: i32, tick: i32) -> u128 {
        const PI_NUM: u128 = 355;
        const PI_DENOM: u128 = 113;
    
        let gamma_sq = gamma * gamma;
        let tick_offset = (tick - mu) as i128;
        let tick_offset_sq = (tick_offset * tick_offset) as u128;
        let denom = gamma_sq + tick_offset_sq;
    
        if denom == 0 {
            return 0;
        }
    
        let num = gamma_sq;
        let base = l0;
    
        // (l0 * gamma^2) / (π * gamma * (gamma^2 + (tick - mu)^2))
        let liquidity = base
            .saturating_mul(num)
            .saturating_mul(PI_DENOM)
            / (PI_NUM.saturating_mul(gamma).saturating_mul(denom));
    
        liquidity
    }

    #[test]
    fn test_cauchy_liquidity_profile_generation() {
        let l0 = 1_000_000_000_000;
        let gamma = 1000;
        let mu = 0;
        let tick_spacing = 100;
        let tick_range = -1000..=1000;

        let mut ticks = vec![];
        let mut active_tick_index = None;
        let mut liquidity = 0u128;
        let mut accumulated = 0u128;

        let tick_iter = tick_range.clone().step_by(tick_spacing as usize);
        let ticks_len = tick_iter.clone().count();

        for (i, tick_index) in tick_iter.clone().enumerate() {
            let liq = cauchy_liquidity_at_tick(l0, gamma, mu, tick_index);
            if i == ticks_len / 2 {
                active_tick_index = Some(i);
                liquidity = liq;
            }
            let delta = if i == 0 {
                liq
            } else {
                let prev_liq = cauchy_liquidity_at_tick(l0, gamma, mu, tick_index - tick_spacing);
                liq.saturating_sub(prev_liq)
            };
            accumulated += delta;
            ticks.push(Tick {
                index: tick_index,
                liquidity_delta: delta as i128,
            });
        }

        if accumulated != 0 {
            ticks.push(Tick {
                index: tick_range.end() + tick_spacing,
                liquidity_delta: -(accumulated as i128),
            });
        }

        let sqrt_ratio = to_sqrt_ratio(0).unwrap();

        let key = NodeKey {
            token0: TOKEN0,
            token1: TOKEN1,
            fee: 10,
            tick_spacing: tick_spacing as u32,
            extension: U256::from(0u8),
        };

        let base_state = BasePoolState {
            active_tick_index,
            liquidity,
            sqrt_ratio,
        };

        let state = SplinePoolState {
            base_pool_state: base_state,
            liquidity_factor: l0,
            total_shares: U256::from(l0),
        };

        let pool = SplinePool::new(key, state, ticks);

        let token_amount = TokenAmount {
            amount: 1_000,
            token: TOKEN0,
        };

        let quote = pool.quote(QuoteParams {
            token_amount,
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        }).expect("Quote failed");

        assert!(quote.calculated_amount > 0);
        assert!(quote.consumed_amount > 0);
    }

    #[test]
    fn test_quote_token1_to_token0() {
        let key = NodeKey {
            token0: TOKEN0,
            token1: TOKEN1,
            fee: FEE_ONE_BPS,
            tick_spacing: 1,
            extension: U256::zero(),
        };

        let ticks = vec![
            Tick { index: 0, liquidity_delta: 1_000_000_000 },
            Tick { index: 1, liquidity_delta: -1_000_000_000 },
        ];

        let base_state = BasePoolState {
            active_tick_index: Some(0),
            liquidity: 1_000_000_000,
            sqrt_ratio: U256([0, 0, 1, 0]), 
        };

        let state = SplinePoolState {
            base_pool_state: base_state,
            liquidity_factor: 1_000_000_000,
            total_shares: U256::from(1_000_000_000u128),
        };

        let pool = SplinePool::new(key, state, ticks);

        let quote = pool.quote(QuoteParams {
            token_amount: TokenAmount { amount: 1_000, token: TOKEN1 },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        }).expect("Quote failed");

        assert_eq!(quote.consumed_amount, 501);
        assert_eq!(quote.calculated_amount, 499);
        assert_eq!(quote.fees_paid, 1);
    }

    #[test]
    fn test_quote_token0_to_token1() {
        let key = NodeKey {
            token0: TOKEN0,
            token1: TOKEN1,
            fee: FEE_ONE_BPS,
            tick_spacing: 1,
            extension: U256::zero(),
        };

        let ticks = vec![
            Tick { index: 0, liquidity_delta: 1_000_000_000 },
            Tick { index: 1, liquidity_delta: -1_000_000_000 },
        ];

        let base_state = BasePoolState {
            active_tick_index: Some(1),
            liquidity: 0,
            sqrt_ratio: to_sqrt_ratio(1).unwrap(), 
        };

        let state = SplinePoolState {
            base_pool_state: base_state,
            liquidity_factor: 1_000_000_000,
            total_shares: U256::from(1_000_000_000u128),
        };

        let pool = SplinePool::new(key, state, ticks);

        let quote = pool.quote(QuoteParams {
            token_amount: TokenAmount { amount: 1_000, token: TOKEN0 },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: (),
        }).expect("Quote failed");

        assert_eq!(quote.consumed_amount, 501);
        assert_eq!(quote.calculated_amount, 499);
        assert_eq!(quote.fees_paid, 1);
    }
}