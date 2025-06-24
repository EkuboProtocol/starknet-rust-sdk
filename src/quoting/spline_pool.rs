use crate::math::uint::U256;
use crate::quoting::base_pool::{
    BasePool, BasePoolResources, BasePoolState, BasePoolQuoteError,
    MAX_TICK_AT_MAX_TICK_SPACING, MIN_TICK_AT_MAX_TICK_SPACING, MAX_TICK_SPACING,
};
use crate::quoting::types::{BlockTimestamp, NodeKey, Pool, Quote, QuoteParams, Tick};
use alloc::vec;
use core::ops::Add;
use num_traits::{ToPrimitive};

#[derive(Clone, Copy)]
pub struct SplinePoolState {
    pub base_pool_state: BasePoolState,
    pub liquidity_factor: u128,
    pub total_shares: u256,
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
    total_shares: u256,
    last_compound_time: u64,
}

impl SplinePool {
    pub fn new(key: NodeKey, state: SplinePoolState, sorted_ticks: Vec<Tick>) -> Self {
        let ticks = if !sorted_ticks.is_empty() {
            sorted_ticks
        } else if state.liquidity_factor > 0 {
            Self::generate_liquidity_updates(&key, state.liquidity_factor, false)
        } else {
            vec![] 
        };
        
        SplinePool {
            base_pool: BasePool::new(
                key,
                state.base_pool_state,
                ticks,
            ),
            liquidity_factor: state.liquidity_factor,
            total_shares: state.liquidity_factor.into(),
            last_compound_time: state.last_compound_time,
        }
    }

    fn generate_liquidity_updates(
        key: &NodeKey,
        liquidity_factor: u128,
        is_negative: bool
    ) -> Vec<Tick> {
        const PI_NUM: U256 = U256([355, 0, 0, 0]); 
        const PI_DENOM: U256 = U256([113, 0, 0, 0]);
        const MIN_TICK: i32 = -88722883;
        const MAX_TICK: i32 = 88722883;
        
        let mu: i32 = 0;              
        let gamma: u64 = 1024;        
        let rho: i32 = 0;            
        let tick_spacing = key.tick_spacing as i32;
        
        let bounds = Self::generate_symmetric_bounds(key.tick_spacing);
        
        let lower_fr = MIN_TICK + (MIN_TICK % tick_spacing).abs();
        let upper_fr = MAX_TICK - (MAX_TICK % tick_spacing).abs();
        
        let mut ticks = Vec::new();
        
        let base_liquidity = Self::get_liquidity_at_tick(
            liquidity_factor, 
            is_negative,
            mu, 
            gamma, 
            rho
        );
        
        ticks.push(Tick {
            index: lower_fr,
            liquidity_delta: if is_negative { -base_liquidity as i128 } else { base_liquidity as i128 },
        });
        
        ticks.push(Tick {
            index: upper_fr,
            liquidity_delta: if is_negative { base_liquidity as i128 } else { -base_liquidity as i128 },
        });
        
        let mut prior_liquidity = 0_i128;
        
        for bound in bounds.iter().rev() {
            let liquidity = Self::get_liquidity_at_tick(
                liquidity_factor,
                is_negative,
                mu,
                gamma,
                bound.0
            ) as i128;
            
            let delta = liquidity - prior_liquidity;
            
            ticks.push(Tick {
                index: bound.0,
                liquidity_delta: delta, 
            });
            
            ticks.push(Tick {
                index: bound.1,
                liquidity_delta: -delta,
            });
            
            prior_liquidity = liquidity;
        }
        
        ticks.sort_by_key(|tick| tick.index);
        ticks
    }
    
    // Returns Cauchy distribution liquidity profile:
    // l(l0, gamma, tick) = (l0 / (pi * gamma)) * (1 / (1 + ((tick - mu) / gamma)^2))
    fn get_liquidity_at_tick(
        liquidity_factor: u128, 
        is_negative: bool,
        mu: i32,
        gamma: u64,
        tick: i32
    ) -> u128 {
        let gamma_u256 = U256::from(gamma as u128);
        let shifted_tick = tick - mu;
        let shifted_tick_mag_256 = U256::from(shifted_tick.abs() as u128);
        
        let denom = gamma_u256 * gamma_u256 + shifted_tick_mag_256 * shifted_tick_mag_256;
        
        let num = gamma_u256 * gamma_u256;
        
        let liquidity_factor_u256 = U256::from(liquidity_factor);
        let l_u256 = (liquidity_factor_u256 * num) / denom;
        
        const PI_NUM_U256: U256 = U256([355, 0, 0, 0]);
        const PI_DENOM_U256: U256 = U256([113, 0, 0, 0]);
        
        let l_scaled_u256 = (l_u256 * PI_DENOM_U256) / (PI_NUM_U256 * gamma_u256);
        
        l_scaled_u256.as_u128()
    }
    
    fn generate_symmetric_bounds(tick_spacing: u32) -> Vec<(i32, i32)> {
        const MIN_TICK: i32 = -88722883;
        const MAX_TICK: i32 = 88722883;
        let ts = tick_spacing as i32;
        
        let s = ts * 10;       
        let res = 4;           
        let tick_start = 0;    
        let tick_max = MAX_TICK; 
        
        let mut bounds = Vec::new();
        let dt = ts;
        
        let mut ticks = (tick_start, tick_start + dt);
        
        let mut seg = s;
        let mut step = s / res as i32;
        
        let mut next = (ticks.0 - seg, ticks.1 + seg);
        
        let mut i = 0;
        while ticks.1 != (tick_max + dt) {
            ticks.0 -= step;
            ticks.1 += step;
            
            if ticks.1 > (tick_max + dt) {
                ticks.1 = tick_max + dt;
                bounds.push(ticks);
                break;
            } 
            else if ticks.1 == next.1 {
                if i > 0 {
                    seg *= 2;
                    step *= 2;
                }
                i += 1;
                next = (next.0 - seg, next.1 + seg);
            }
            
            bounds.push(ticks);
        }
        
        bounds
    }
}

impl Pool for SplinePool {
    type Resources = SplinePoolResources;
    type State = SplinePoolState;
    type QuoteError = BasePoolQuoteError;
    type Meta = BlockTimestamp;

    fn get_key(&self) -> &NodeKey {
        self.base_pool.get_key()
    }

    fn get_state(&self) -> Self::State {
        SplinePoolState {
            base_pool_state: self.base_pool.get_state(),
            liquidity_factor: self.liquidity_factor,
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
    use crate::math::tick::to_sqrt_ratio;
    use crate::math::uint::U256;
    use crate::quoting::types::{QuoteParams, TokenAmount};

    const TOKEN0: U256 = U256([1, 0, 0, 0]);
    const TOKEN1: U256 = U256([2, 0, 0, 0]);
    const EXTENSION: U256 = U256([3, 0, 0, 0]);
    const LIQUIDITY: u128 = 1_000_000;
    const INVALID_TOKEN: U256 = U256([999, 0, 0, 0]);

    fn default_pool() -> SplinePool {
        SplinePool::new(TOKEN0, TOKEN1, EXTENSION, to_sqrt_ratio(0).unwrap(), LIQUIDITY)
    }

    mod constructor_validation {
        use super::*;

        #[test]
        fn test_zero_liquidity_pool() {
            let pool = SplinePool::new(TOKEN0, TOKEN1, EXTENSION, to_sqrt_ratio(0).unwrap(), 0);
            assert!(!pool.has_liquidity());
        
            let params = QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: 0,
            };
        
            let result = pool.quote(params).expect("Quote should succeed");
            assert_eq!(result.calculated_amount, 0);
            assert_eq!(result.consumed_amount, 0);
        }

        #[test]
        fn test_min_price_constructor() {
            let min_price_pool = SplinePool::new(
                TOKEN0, 
                TOKEN1, 
                EXTENSION, 
                to_sqrt_ratio(MIN_TICK_AT_MAX_TICK_SPACING).unwrap(), 
                LIQUIDITY
            );
            
            assert!(min_price_pool.has_liquidity());
            assert_eq!(
                min_price_pool.get_state().base_pool_state.sqrt_ratio, 
                to_sqrt_ratio(MIN_TICK_AT_MAX_TICK_SPACING).unwrap()
            );

            let min_params = QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: 0,
            };
            
            min_price_pool.quote(min_params).expect("Quote at min price failed");
        }

        #[test]
        fn test_max_price_constructor() {
            let max_price_pool = SplinePool::new(
                TOKEN0, 
                TOKEN1, 
                EXTENSION, 
                to_sqrt_ratio(MAX_TICK_AT_MAX_TICK_SPACING).unwrap(), 
                LIQUIDITY
            );
            
            assert!(max_price_pool.has_liquidity());
            assert_eq!(
                max_price_pool.get_state().base_pool_state.sqrt_ratio, 
                to_sqrt_ratio(MAX_TICK_AT_MAX_TICK_SPACING).unwrap()
            );

            let max_params = QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN1,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: 0,
            };
            
            max_price_pool.quote(max_params).expect("Quote at max price failed");
        }

        #[test]
        fn test_min_liquidity_constructor() {
            let min_liquidity: u128 = 1;
            let pool = SplinePool::new(
                TOKEN0, TOKEN1, EXTENSION, to_sqrt_ratio(0).unwrap(), min_liquidity
            );
            
            assert!(pool.has_liquidity());
            assert_eq!(pool.get_state().liquidity_factor, min_liquidity);
            
            let params = QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: 0,
            };
            
            pool.quote(params).expect("Min liquidity quote failed");
        }

        #[test]
        fn test_max_liquidity_constructor() {
            let max_liquidity: u128 = i128::MAX as u128;
            let pool = SplinePool::new(
                TOKEN0, TOKEN1, EXTENSION, to_sqrt_ratio(0).unwrap(), max_liquidity
            );
            
            assert!(pool.has_liquidity());
            assert_eq!(pool.get_state().liquidity_factor, max_liquidity);
            
            let params = QuoteParams {
                token_amount: TokenAmount {
                    amount: 1000,
                    token: TOKEN0,
                },
                sqrt_ratio_limit: None,
                override_state: None,
                meta: 0,
            };
            
            pool.quote(params).expect("Max liquidity quote failed");
        }
    }
    
    #[test]
    fn test_quote_token0_input() {
        let pool = default_pool();
        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: 1234,
        };

        let quote = pool.quote(params).expect("Quote failed");
        assert_eq!(quote.consumed_amount, 1000);
        assert!(quote.calculated_amount > 0);
        assert_eq!(quote.state_after.liquidity_factor, LIQUIDITY);
    }

    #[test]
    fn test_quote_token1_input() {
        let pool = default_pool();
        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: 5678,
        };

        let quote = pool.quote(params).expect("Quote failed");
        assert_eq!(quote.consumed_amount, 1000);
        assert!(quote.calculated_amount > 0);
        assert_eq!(quote.state_after.liquidity_factor, LIQUIDITY);
    }

    #[test]
    fn test_liquidity_factor_impact() {
        let low_liquidity_pool = SplinePool::new(
            TOKEN0, TOKEN1, EXTENSION, to_sqrt_ratio(0).unwrap(), LIQUIDITY / 10
        );
        
        let high_liquidity_pool = SplinePool::new(
            TOKEN0, TOKEN1, EXTENSION, to_sqrt_ratio(0).unwrap(), LIQUIDITY * 10
        );
        
        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 10000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: 0,
        };
        
        let default_quote = default_pool().quote(params.clone()).expect("Default quote failed");
        let low_quote = low_liquidity_pool.quote(params.clone()).expect("Low liquidity quote failed");
        let high_quote = high_liquidity_pool.quote(params.clone()).expect("High liquidity quote failed");
        
        assert_ne!(low_quote.state_after.base_pool_state.sqrt_ratio, default_quote.state_after.base_pool_state.sqrt_ratio);
        assert_ne!(default_quote.state_after.base_pool_state.sqrt_ratio, high_quote.state_after.base_pool_state.sqrt_ratio);
        assert_ne!(low_quote.state_after.base_pool_state.sqrt_ratio, high_quote.state_after.base_pool_state.sqrt_ratio);
    }

    #[test]
    fn test_quote_with_override_state() {
        let pool = default_pool();
        let original_state = pool.get_state();

        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 500,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: Some(original_state),
            meta: 777,
        };

        let quote = pool.quote(params).expect("Override quote failed");
        assert_eq!(quote.state_after.liquidity_factor, LIQUIDITY);
    }

    #[test]
    fn test_consecutive_quotes_state() {
        let pool = default_pool();
        let initial_state = pool.get_state();
        
        let params1 = QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: 0,
        };
        
        let quote1 = pool.quote(params1).expect("First quote failed");
        let intermediate_state = quote1.state_after;
        
        let params2 = QuoteParams {
            token_amount: TokenAmount {
                amount: 2000,
                token: TOKEN1,
            },
            sqrt_ratio_limit: None,
            override_state: Some(intermediate_state),
            meta: 0,
        };
        
        let quote2 = pool.quote(params2).expect("Second quote failed");
        let final_state = quote2.state_after;
        
        assert_ne!(initial_state.base_pool_state.sqrt_ratio, intermediate_state.base_pool_state.sqrt_ratio);
        assert_ne!(intermediate_state.base_pool_state.sqrt_ratio, final_state.base_pool_state.sqrt_ratio);
        assert_eq!(initial_state.liquidity_factor, intermediate_state.liquidity_factor);
        assert_eq!(intermediate_state.liquidity_factor, final_state.liquidity_factor);
    }

    #[test]
    fn test_invalid_token_quote() {
        let pool = default_pool();
        let params = QuoteParams {
            token_amount: TokenAmount {
                amount: 1000,
                token: INVALID_TOKEN,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: 0,
        };
        
        let result = pool.quote(params);
        assert!(result.is_err(), "Quote with invalid token should fail");
        
        if let Err(error) = result {
            match error {
                BasePoolQuoteError::InvalidToken => {},
                _ => panic!("Unexpected error type: quote with invalid token should return InvalidToken"),
            }
        }
    }
}