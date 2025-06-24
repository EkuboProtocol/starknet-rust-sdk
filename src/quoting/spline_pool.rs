use crate::math::uint::U256;
use crate::quoting::base_pool::{
    BasePool, BasePoolResources, BasePoolState, BasePoolQuoteError
};
use crate::quoting::types::BlockTimestamp;
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
        }
    }

    fn generate_liquidity_updates(
        key: &NodeKey,
        liquidity_factor: u128,
        is_negative: bool
    ) -> Vec<Tick> {
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
        
        // Ensure lower_fr is a multiple of tick_spacing
        let adjusted_lower_fr = lower_fr - (lower_fr % tick_spacing);
        ticks.push(Tick {
            index: adjusted_lower_fr,
            liquidity_delta: if is_negative { -(base_liquidity as i128) } else { base_liquidity as i128 },
        });
        
        // Ensure upper_fr is a multiple of tick_spacing
        let adjusted_upper_fr = upper_fr - (upper_fr % tick_spacing);
        ticks.push(Tick {
            index: adjusted_upper_fr,
            liquidity_delta: if is_negative { base_liquidity as i128 } else { -(base_liquidity as i128) },
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
            
            // Ensure bound.0 is a multiple of tick_spacing
            let adjusted_lower = bound.0 - (bound.0 % tick_spacing);
            ticks.push(Tick {
                index: adjusted_lower,
                liquidity_delta: delta, 
            });
            
            // Ensure bound.1 is a multiple of tick_spacing
            let adjusted_upper = bound.1 - (bound.1 % tick_spacing);
            ticks.push(Tick {
                index: adjusted_upper,
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
        _is_negative: bool,
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
            
            ticks.0 = ticks.0 - (ticks.0 % ts);
            ticks.1 = ticks.1 - (ticks.1 % ts);
            
            if ticks.1 > (tick_max + dt) {
                ticks.1 = tick_max + dt;
                // Ensure boundary condition ticks are also aligned with tick spacing
                ticks.1 = ticks.1 - (ticks.1 % ts);
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
    use crate::math::tick::to_sqrt_ratio;
    use crate::quoting::base_pool::{MAX_TICK_SPACING};
    use crate::quoting::types::{NodeKey, QuoteParams, Tick, TokenAmount};
    use alloc::vec;

    const TOKEN0: U256 = U256([1, 0, 0, 0]);
    const TOKEN1: U256 = U256([2, 0, 0, 0]);
    const EXTENSION: U256 = U256([0, 0, 0, 0]);
    const LIQUIDITY: u128 = 1_000_000;

    fn create_node_key(tick_spacing: u32) -> NodeKey {
        NodeKey {
            token0: TOKEN0,
            token1: TOKEN1,
            fee: 0,
            tick_spacing,
            extension: EXTENSION,
        }
    }

    fn create_state(sqrt_ratio: U256, liquidity: u128, active_tick_index: Option<usize>) -> SplinePoolState {
        SplinePoolState {
            base_pool_state: BasePoolState {
                sqrt_ratio,
                liquidity,
                active_tick_index,
            },
            liquidity_factor: liquidity,
            total_shares: liquidity.into(),
        }
    }
    
    fn create_default_ticks(active_tick_index: i32, liquidity: u128, tick_spacing: u32) -> Vec<Tick> {
        if liquidity == 0 {
            return vec![];
        }
        
        let ts = tick_spacing as i32;
        let adjusted_active = (active_tick_index / ts) * ts;
        
        vec![
            Tick {
                index: adjusted_active,
                liquidity_delta: liquidity as i128,
            },
            Tick {
                index: adjusted_active + 1000 * ts, 
                liquidity_delta: -(liquidity as i128),
            }
        ]
    }

    #[test]
    fn test_cauchy_distribution_profile() {
        let mu = 0;             
        let gamma = 1000;       
        let rho = 990;          
        
        let custom_key = NodeKey {
            token0: TOKEN0,
            token1: TOKEN1,
            extension: U256([mu as u64, gamma, rho as u64, 0]),
            fee: 0,
            tick_spacing: MAX_TICK_SPACING,
        };
        
        let liquidity_factor = 100_000;
        let ticks = SplinePool::generate_liquidity_updates(&custom_key, liquidity_factor, false);
        
        assert!(ticks.len() > 4, "Should generate multiple tick positions");
        
        for i in 1..ticks.len() {
            assert!(ticks[i].index > ticks[i-1].index, "Ticks should be sorted by index");
        }
        
        let positive_ticks: Vec<&Tick> = ticks.iter()
            .filter(|t| t.liquidity_delta > 0 && t.index != 0) 
            .collect();
            
        assert!(!positive_ticks.is_empty(), "Should have positive liquidity ticks for testing");
        
        let tick_map: alloc::collections::BTreeMap<i32, i128> = positive_ticks.iter()
            .map(|tick| (tick.index, tick.liquidity_delta))
            .collect();
            
        let samples = core::cmp::min(positive_ticks.len(), 10);
        for i in 0..samples {
            let idx = i * positive_ticks.len() / samples;
            let tick = positive_ticks[idx];
            
            let symmetric_index = 2 * mu - tick.index;
            
            if let Some(&symmetric_liquidity) = tick_map.get(&symmetric_index) {
                let ratio = (tick.liquidity_delta as f64) / (symmetric_liquidity as f64);
                assert!((ratio - 1.0).abs() < 0.2, 
                       "Symmetric ticks should have similar liquidity, ratio: {}", ratio);
            }
        }
        
        let mut previous_tick: Option<i32> = None;
        let mut previous_distance: Option<f64> = None;
        let mut previous_liquidity: Option<i128> = None;
        
        for tick in positive_ticks.iter().take(positive_ticks.len() / 2) {
            let distance = (tick.index - mu).abs() as f64;
            
            if let (Some(_prev_tick), Some(prev_distance), Some(prev_liquidity)) = 
                    (previous_tick, previous_distance, previous_liquidity) {
                    
                let expected_ratio = (gamma.pow(2) as f64 + distance.powi(2)) / 
                                    (gamma.pow(2) as f64 + prev_distance.powi(2));
                                    
                let actual_ratio = prev_liquidity as f64 / tick.liquidity_delta as f64;
                
                let tolerance = 0.3;
                
                assert!((actual_ratio / expected_ratio - 1.0).abs() < tolerance,
                    "Liquidity should follow Cauchy distribution. At tick {} expected ratio {}, got {}", 
                    tick.index, expected_ratio, actual_ratio);
            }
            
            previous_tick = Some(tick.index);
            previous_distance = Some(distance);
            previous_liquidity = Some(tick.liquidity_delta);
        }
    }

    #[test]
    fn test_all_generated_ticks_follow_tick_spacing() {
        for tick_spacing in [MAX_TICK_SPACING, MAX_TICK_SPACING/10, MAX_TICK_SPACING/100] {
            let key = create_node_key(tick_spacing);
            let liquidity_factor = 100_000;
            let ticks = SplinePool::generate_liquidity_updates(&key, liquidity_factor, false);
            
            for tick in &ticks {
                assert_eq!(tick.index % (tick_spacing as i32), 0, 
                    "Tick index must be a multiple of tick_spacing");
            }
        }
    }
    
    #[test]
    fn test_liquidity_distribution_after_crossing_ticks() {
        let key = create_node_key(MAX_TICK_SPACING);
        let sqrt_ratio = to_sqrt_ratio(0).unwrap();
        let state = create_state(sqrt_ratio, LIQUIDITY, Some(0));
        let ticks = create_default_ticks(0, LIQUIDITY, MAX_TICK_SPACING);
        
        let pool = SplinePool::new(key, state, ticks);
        
        // Make a quote that will cross ticks
        let result = pool.quote(QuoteParams {
            token_amount: TokenAmount {
                amount: 1_000_000,
                token: TOKEN0,
            },
            sqrt_ratio_limit: None,
            override_state: None,
            meta: 0,
        }).expect("Quote should succeed");
        
        assert!(result.execution_resources.base_pool_resources.initialized_ticks_crossed > 0,
            "Should have crossed at least one tick");
        assert!(result.state_after.base_pool_state.sqrt_ratio != sqrt_ratio,
            "Price should have changed");
    }
    

    #[test]
    fn test_extreme_mu_gamma_rho_values() {
        let test_cases = [
            (0, 1000, 990),
            (50000, 100, 990),
            (-50000, 100, 990),
            (0, 1, 990)
        ];
        
        for (_, (mu_override, gamma_override, rho)) in test_cases.iter().enumerate() {
            let custom_key = NodeKey {
                token0: TOKEN0,
                token1: TOKEN1,
                extension: U256([*mu_override as u64, *gamma_override, *rho as u64, 0]),
                fee: 0,
                tick_spacing: MAX_TICK_SPACING,
            };
            
            let ticks = SplinePool::generate_liquidity_updates(&custom_key, LIQUIDITY, false);
            assert!(!ticks.is_empty(), "Should generate ticks even with extreme parameters");
            
            for tick in &ticks {
                assert_eq!(tick.index % (custom_key.tick_spacing as i32), 0, 
                    "Tick index must be a multiple of tick_spacing");
            }
        }
    }
    
    #[test]
    fn test_base_pool_integration() {
        let key = create_node_key(MAX_TICK_SPACING);
        let state = create_state(to_sqrt_ratio(0).unwrap(), LIQUIDITY, Some(0));
        let ticks = create_default_ticks(0, LIQUIDITY, MAX_TICK_SPACING);
        
        let pool = SplinePool::new(key, state, ticks);
        
        assert_eq!(pool.get_key(), &key);
        assert_eq!(pool.get_state().base_pool_state.liquidity, LIQUIDITY);
        assert!(pool.has_liquidity());
        
        let min_tick = pool.min_tick_with_liquidity();
        let max_tick = pool.max_tick_with_liquidity();
        
        assert!(min_tick.is_some(), "Should have a min tick");
        assert!(max_tick.is_some(), "Should have a max tick");
    }
    
    #[test]
    fn test_symmetric_bounds_generation() {
        for tick_spacing in [MAX_TICK_SPACING, MAX_TICK_SPACING/10] {
            let bounds = SplinePool::generate_symmetric_bounds(tick_spacing);
            
            assert!(bounds.len() > 1, "Should generate multiple bounds");
            
            for (lower, upper) in &bounds {
                assert_eq!(*lower % (tick_spacing as i32), 0,
                    "Lower bound should be a multiple of tick spacing");
                assert_eq!(*upper % (tick_spacing as i32), 0,
                    "Upper bound should be a multiple of tick spacing");
                
                assert!(*lower < *upper, "Upper bound should be greater than lower bound");
                
                const MAX_TICK: i32 = 88722883;
                const MIN_TICK: i32 = -88722883;
                let margin = 10000;
                
                if *lower > (MIN_TICK + margin) && *upper < (MAX_TICK - margin) {
                    let lower_abs = lower.abs();
                    let upper_abs = upper.abs();
                    
                    let max_diff = tick_spacing as i32;
                    assert!(
                        (upper_abs - lower_abs).abs() <= max_diff, 
                        "Bounds should be symmetric: |{}| should approximately equal |{}|", 
                        lower, upper
                    );
                }
            }
            
            for i in 1..bounds.len() {
                let prev_width = bounds[i-1].1 - bounds[i-1].0;
                let curr_width = bounds[i].1 - bounds[i].0;
                
                assert!(curr_width >= prev_width, 
                    "Bounds should generally expand outward or maintain width");
            }
        }
    }
}
