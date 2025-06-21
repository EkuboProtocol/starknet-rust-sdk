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
    pub fn new(
        token0: U256,
        token1: U256,
        extension: U256,
        sqrt_ratio: U256,
        liquidity_factor: u128,
        last_compound_time: u64,
    ) -> Self {
        let signed_liquidity: i128 =
            liquidity_factor.to_i128().expect("liquidity_factor exceeds i128");

        let ticks = if liquidity_factor > 0 {
            vec![
                Tick {
                    index: MIN_TICK_AT_MAX_TICK_SPACING,
                    liquidity_delta: signed_liquidity,
                },
                Tick {
                    index: MAX_TICK_AT_MAX_TICK_SPACING,
                    liquidity_delta: -signed_liquidity,
                },
            ]
        } else {
            vec![] // Empty ticks for zero liquidity
        };

        SplinePool {
            base_pool: BasePool::new(
                NodeKey {
                    token0,
                    token1,
                    fee: 0,
                    tick_spacing: MAX_TICK_SPACING,
                    extension,
                },
                BasePoolState {
                    sqrt_ratio,
                    liquidity: liquidity_factor.into(),
                    active_tick_index: if liquidity_factor > 0 {
                        Some(0)
                    } else {
                        None
                    },
                },
                ticks,
            ),
            liquidity_factor,
            total_shares: liquidity_factor.into(),
            last_compound_time,
        }
    }

    fn calculate_shares(&self, total_shares: u256, factor: u128, total_factor: u128) -> u256 {
        if total_factor == 0 {
            return factor.into(); // First liquidity addition
        }
        let denom: u256 = total_factor.into();
        let num: u256 = factor.into();
        (total_shares * num) / denom
    }
    
    fn calculate_factor(&self, total_factor: u128, shares: u256, total_shares: u256) -> u128 {
        if total_shares == 0.into() {
            return 0; // No shares exist
        }
        let total_factor_u256: u256 = total_factor.into();
        let factor_u256 = (total_factor_u256 * shares) / total_shares;
        factor_u256.try_into().unwrap()
    }

    pub fn compound_fees(&mut self, block_time: u64) -> u128 {
        if block_time <= self.last_compound_time {
            return 0; // No time has passed, no fees to compound
        }
        
        // In the real contract, this would calculate and add fees
        // For simulation, we just track that compounding has occurred
        // by updating the timestamp
        let liquidity_fees = 0; // No actual fee calculation in the simplified model
        self.last_compound_time = block_time;
        
        liquidity_fees
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
        let block_time = params.meta;
        
        // Check if fee compounding would happen in this quote
        let should_compound = params.override_state.is_none() && block_time > self.last_compound_time;
        let fee_compounds = if should_compound { 1 } else { 0 };
        
        // Use override state or current state for liquidity factor
        // We don't simulate actual fee additions in this simplified model
        let liquidity_factor = params.override_state.map_or(self.liquidity_factor, |os| os.liquidity_factor);
        
        // Delegate actual swap calculation to base_pool
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
                total_shares: self.total_shares, // Shares don't change during quote
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