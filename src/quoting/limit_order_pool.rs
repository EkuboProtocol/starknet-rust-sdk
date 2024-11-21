use crate::math::swap::is_price_increasing;
use crate::math::tick::{to_sqrt_ratio, MAX_SQRT_RATIO, MIN_SQRT_RATIO};
use crate::math::uint::U256;
use crate::quoting::base_pool::{BasePool, BasePoolQuoteError, BasePoolResources, BasePoolState};
use crate::quoting::types::{NodeKey, Pool, Quote, QuoteParams, Tick};
use crate::quoting::util::find_nearest_initialized_tick_index;
use alloc::vec::Vec;
use core::ops::Add;
use num_traits::Zero;

use super::types::TokenAmount;
use super::util::approximate_number_of_tick_spacings_crossed;

#[derive(Clone, Copy)]
pub struct LimitOrderPoolState {
    pub base_pool_state: BasePoolState,

    // the minimum and maximum active tick index we've reached since the sorted ticks were last updated.
    // if None, then we haven't seen any swap since the last sorted ticks update.
    // if the minimum active tick index is Some(None), then we have swapped through all the ticks less than the current active tick index
    // if the maxmimum active tick index is Some(sorted_ticks.len() - 1), then we have swapped through all the ticks greater than the current active tick index
    pub tick_indices_reached: Option<(Option<usize>, Option<usize>)>,
}

#[derive(Default, Clone, Copy)]
pub struct LimitOrderPoolResources {
    pub base_pool_resources: BasePoolResources,
    // the number of orders that were pulled, i.e. the number of times we crossed active ticks
    //   that were the end boundary of a position
    pub orders_pulled: u32,
}

impl Add for LimitOrderPoolResources {
    type Output = LimitOrderPoolResources;

    fn add(self, rhs: Self) -> Self::Output {
        LimitOrderPoolResources {
            base_pool_resources: self.base_pool_resources + rhs.base_pool_resources,
            orders_pulled: self.orders_pulled + rhs.orders_pulled,
        }
    }
}

pub struct LimitOrderPool {
    base_pool: BasePool,
}

pub const LIMIT_ORDER_TICK_SPACING: i32 = 128;
pub const DOUBLE_LIMIT_ORDER_TICK_SPACING: i32 = 2i32 * LIMIT_ORDER_TICK_SPACING;

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
                    tick_spacing: LIMIT_ORDER_TICK_SPACING.unsigned_abs(),
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
            tick_indices_reached: None,
        }
    }

    fn quote(
        &self,
        params: QuoteParams<Self::State, Self::Meta>,
    ) -> Result<Quote<Self::Resources, Self::State>, Self::QuoteError> {
        let initial_state = params.override_state.unwrap_or_else(|| self.get_state());

        let mut calculated_amount = 0;
        let mut consumed_amount = 0;
        let mut fees_paid = 0;
        let mut base_pool_resources = BasePoolResources::default();
        let mut base_pool_state = initial_state.base_pool_state;

        let is_increasing = is_price_increasing(
            params.token_amount.amount,
            params.token_amount.token == self.get_key().token1,
        );

        let sorted_ticks = self.base_pool.get_sorted_ticks();

        let next_unpulled_order_tick_index =
            initial_state.tick_indices_reached.map(|(lower, upper)| {
                if lower == upper {
                    lower // we haven't moved out of the active tick, so the current tick is unpulled
                } else if is_increasing {
                    upper.map(|upper| {
                        Ord::min(
                            sorted_ticks.len() - 1, // will not overflow because upper is Some
                            upper + 1,
                        )
                    })
                } else {
                    lower.and_then(|lower| lower.checked_sub(1))
                }
            });

        if let Some(next_unpulled_order_tick_index) = next_unpulled_order_tick_index {
            let active_tick_index = base_pool_state.active_tick_index;

            let active_tick_sqrt_ratio_limit = if is_increasing {
                active_tick_index
                    .map_or_else(|| sorted_ticks.first(), |idx| sorted_ticks.get(idx + 1))
                    .map_or(Ok(MAX_SQRT_RATIO), |next| {
                        to_sqrt_ratio(next.index).ok_or(BasePoolQuoteError::InvalidTick(next.index))
                    })
            } else {
                active_tick_index.map_or(Ok(MIN_SQRT_RATIO), |idx| {
                    let tick = sorted_ticks[idx]; // is always valid
                    to_sqrt_ratio(tick.index).ok_or(BasePoolQuoteError::InvalidTick(tick.index))
                })
            }?;

            let params_sqrt_ratio_limit = params.sqrt_ratio_limit.unwrap_or(if is_increasing {
                MAX_SQRT_RATIO
            } else {
                MIN_SQRT_RATIO
            });

            let active_tick_boundary_sqrt_ratio = if is_increasing {
                Ord::min(active_tick_sqrt_ratio_limit, params_sqrt_ratio_limit)
            } else {
                Ord::max(active_tick_sqrt_ratio_limit, params_sqrt_ratio_limit)
            };

            // swap to (at most) the boundary of the current active tick
            let quote_to_active_tick_boundary = self.base_pool.quote(QuoteParams {
                sqrt_ratio_limit: Some(active_tick_boundary_sqrt_ratio),
                token_amount: params.token_amount,
                override_state: Some(base_pool_state),
                meta: (),
            })?;

            calculated_amount += quote_to_active_tick_boundary.calculated_amount;
            consumed_amount += quote_to_active_tick_boundary.consumed_amount;
            fees_paid += quote_to_active_tick_boundary.fees_paid;
            base_pool_resources += quote_to_active_tick_boundary.execution_resources;

            base_pool_state = quote_to_active_tick_boundary.state_after;

            let amount_remaining = params.token_amount.amount - consumed_amount;

            // skip the range of pulled orders and start from the sqrt ratio of the next unpulled order if we have some amount remaining
            if !amount_remaining.is_zero() {
                let skip_starting_sqrt_ratio = if is_increasing {
                    let next_unpulled_order_sqrt_ratio = next_unpulled_order_tick_index.map_or(
                        Ok(MAX_SQRT_RATIO), // note that reaching this case implies that the pool has no initialized ticks
                        |idx| {
                            let tick_index = sorted_ticks[idx].index;
                            to_sqrt_ratio(tick_index)
                                .ok_or(BasePoolQuoteError::InvalidTick(tick_index))
                        },
                    )?;

                    next_unpulled_order_sqrt_ratio.clamp(
                        base_pool_state.sqrt_ratio, // for the case that next_unpulled_order_index refers to the starting tick index which we've already crossed
                        params_sqrt_ratio_limit,
                    )
                } else {
                    let next_unpulled_order_sqrt_ratio =
                        next_unpulled_order_tick_index.map_or(Ok(MIN_SQRT_RATIO), |idx| {
                            sorted_ticks
                                .get(idx + 1)
                                .map(|tick| {
                                    to_sqrt_ratio(tick.index)
                                        .ok_or(BasePoolQuoteError::InvalidTick(tick.index))
                                })
                                .unwrap_or(Ok(MAX_SQRT_RATIO))
                        })?;

                    next_unpulled_order_sqrt_ratio.clamp(
                        params_sqrt_ratio_limit,
                        base_pool_state.sqrt_ratio, // for the case that next_unpulled_order_index refers to the starting tick index which we've already crossed
                    )
                };

                // account for the tick spacings of the uninitialized ticks that we will skip next
                base_pool_resources.tick_spacings_crossed +=
                    approximate_number_of_tick_spacings_crossed(
                        base_pool_state.sqrt_ratio,
                        skip_starting_sqrt_ratio,
                        LIMIT_ORDER_TICK_SPACING.unsigned_abs(),
                    );

                let liquidity_at_next_unpulled_order_tick_index = {
                    let mut current_liquidity = base_pool_state.liquidity;
                    let mut current_active_tick_index = base_pool_state.active_tick_index;

                    // apply all liquidity_deltas in between the current active tick and the next unpulled order
                    loop {
                        let (next_tick_index, liquidity_delta) = if is_increasing
                            && current_active_tick_index < next_unpulled_order_tick_index
                        {
                            let next_tick_index = current_active_tick_index.map_or(
                                0,
                                |idx| idx + 1, // valid index because next_unpulled_order_tick_index is larger than current_active_tick_index
                            );

                            (
                                Some(next_tick_index),
                                sorted_ticks[next_tick_index].liquidity_delta,
                            )
                        } else if !is_increasing
                            && current_active_tick_index > next_unpulled_order_tick_index
                        {
                            let current_tick_index = current_active_tick_index.unwrap(); // safe because current_active_tick_index is larger than next_unpulled_order_tick_index

                            (
                                current_tick_index.checked_sub(1),
                                sorted_ticks[current_tick_index].liquidity_delta,
                            )
                        } else {
                            break;
                        };

                        current_active_tick_index = next_tick_index;

                        if liquidity_delta.is_positive() == is_increasing {
                            current_liquidity += liquidity_delta.unsigned_abs();
                        } else {
                            current_liquidity -= liquidity_delta.unsigned_abs();
                        };
                    }

                    current_liquidity
                };

                let quote_from_next_unpulled_order = self.base_pool.quote(QuoteParams {
                    sqrt_ratio_limit: params.sqrt_ratio_limit,
                    token_amount: TokenAmount {
                        amount: amount_remaining,
                        token: params.token_amount.token,
                    },
                    override_state: Some(BasePoolState {
                        active_tick_index: next_unpulled_order_tick_index,
                        sqrt_ratio: skip_starting_sqrt_ratio,
                        liquidity: liquidity_at_next_unpulled_order_tick_index,
                    }),
                    meta: (),
                })?;

                calculated_amount += quote_from_next_unpulled_order.calculated_amount;
                consumed_amount += quote_from_next_unpulled_order.consumed_amount;
                fees_paid += quote_from_next_unpulled_order.fees_paid;
                base_pool_resources += quote_from_next_unpulled_order.execution_resources;

                base_pool_state = quote_from_next_unpulled_order.state_after;
            }
        } else {
            let quote_simple = self.base_pool.quote(QuoteParams {
                sqrt_ratio_limit: params.sqrt_ratio_limit,
                override_state: Some(initial_state.base_pool_state),
                token_amount: params.token_amount,
                meta: (),
            })?;

            calculated_amount += quote_simple.calculated_amount;
            consumed_amount += quote_simple.consumed_amount;
            fees_paid += quote_simple.fees_paid;
            base_pool_resources += quote_simple.execution_resources;

            base_pool_state = quote_simple.state_after;
        }

        let (tick_index_before, tick_index_after) = (
            initial_state.base_pool_state.active_tick_index,
            base_pool_state.active_tick_index,
        );

        let new_tick_indices_reached = if is_increasing {
            initial_state
                .tick_indices_reached
                .map_or((tick_index_before, tick_index_after), |(lower, upper)| {
                    (lower, Ord::max(upper, tick_index_after))
                })
        } else {
            initial_state
                .tick_indices_reached
                .map_or((tick_index_after, tick_index_before), |(lower, upper)| {
                    (Ord::min(lower, tick_index_after), upper)
                })
        };

        let from_tick_index = next_unpulled_order_tick_index
            .unwrap_or(initial_state.base_pool_state.active_tick_index);
        let to_tick_index = if is_increasing {
            Ord::max(from_tick_index, tick_index_after)
        } else {
            Ord::min(from_tick_index, tick_index_after)
        };

        let orders_pulled =
            calculate_orders_pulled(from_tick_index, to_tick_index, is_increasing, sorted_ticks);

        Ok(Quote {
            calculated_amount,
            consumed_amount,
            execution_resources: LimitOrderPoolResources {
                base_pool_resources,
                orders_pulled,
            },
            fees_paid,
            is_price_increasing: is_increasing,
            state_after: LimitOrderPoolState {
                base_pool_state,
                tick_indices_reached: Some(new_tick_indices_reached),
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

fn calculate_orders_pulled(
    from: Option<usize>,
    to: Option<usize>,
    is_increasing: bool,
    sorted_ticks: &Vec<Tick>,
) -> u32 {
    let mut current = from;
    let mut orders_pulled = 0;

    while current != to {
        let crossed_tick = if is_increasing {
            let crossed_tick = current.map_or(0, |idx| idx + 1);
            current = Some(crossed_tick);
            crossed_tick
        } else {
            let crossed_tick = current.unwrap();
            current = crossed_tick.checked_sub(1);
            crossed_tick
        };

        if !(sorted_ticks[crossed_tick].index.unsigned_abs()
            % DOUBLE_LIMIT_ORDER_TICK_SPACING.unsigned_abs())
        .is_zero()
        {
            orders_pulled += 1;
        }
    }

    orders_pulled
}

#[cfg(test)]
mod tests {
    use crate::math::tick::to_sqrt_ratio;
    use crate::math::uint::U256;
    use crate::quoting::limit_order_pool::{LimitOrderPool, LIMIT_ORDER_TICK_SPACING};
    use crate::quoting::types::{Pool, QuoteParams, Tick, TokenAmount};
    use alloc::vec;

    const TOKEN0: U256 = U256([0, 0, 0, 1]);
    const TOKEN1: U256 = U256([0, 0, 0, 2]);
    const EXTENSION: U256 = U256([0, 0, 0, 3]);

    #[test]
    fn test_swap_one_for_zero_partial() {
        let liquidity: i128 = 10000000;
        let pool = LimitOrderPool::new(
            TOKEN0,
            TOKEN1,
            EXTENSION,
            to_sqrt_ratio(0).unwrap(),
            0,
            liquidity.unsigned_abs(),
            vec![
                Tick {
                    index: 0,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING,
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
                    token: TOKEN1,
                    amount: 10000,
                },
            })
            .expect("Quote failed");

        assert_eq!(quote.fees_paid, 0);
        assert_eq!(
            quote.state_after.tick_indices_reached,
            Some((Some(0), Some(1)))
        );
        assert_eq!(quote.consumed_amount, 641);
        assert_eq!(quote.calculated_amount, 639);
        assert_eq!(quote.execution_resources.orders_pulled, 1);
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
            TOKEN0,
            TOKEN1,
            EXTENSION,
            to_sqrt_ratio(0).unwrap(),
            0,
            liquidity.unsigned_abs(),
            vec![
                Tick {
                    index: 0,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING,
                    liquidity_delta: -liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING * 2,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING * 3,
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
                    token: TOKEN1,
                    amount: 1000,
                },
            })
            .expect("Quote failed");

        assert_eq!(quote.fees_paid, 0);
        assert_eq!(
            quote.state_after.tick_indices_reached,
            Some((Some(0), Some(2)))
        );
        assert_eq!(quote.consumed_amount, 1000);
        assert_eq!(quote.calculated_amount, 997);
        assert_eq!(quote.execution_resources.orders_pulled, 1);
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

    #[test]
    fn test_order_sell_token0_for_token1_can_only_be_executed_once() {
        let liquidity: i128 = 10000000;
        let pool = LimitOrderPool::new(
            TOKEN0,
            TOKEN1,
            EXTENSION,
            to_sqrt_ratio(0).unwrap(),
            0,
            liquidity.unsigned_abs(),
            vec![
                Tick {
                    index: 0,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING,
                    liquidity_delta: -liquidity,
                },
            ],
        );

        // trade all the way through the order
        let quote0 = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING),
                override_state: None,
                meta: (),
                token_amount: TokenAmount {
                    token: TOKEN1,
                    amount: 1000,
                },
            })
            .expect("quote0 failed");

        assert_eq!(
            quote0.state_after.base_pool_state.active_tick_index,
            Some(1)
        );
        assert_eq!(
            quote0.state_after.base_pool_state.sqrt_ratio,
            to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING).unwrap()
        );
        assert_eq!(
            quote0.state_after.tick_indices_reached,
            Some((Some(0), Some(1)))
        );
        assert_eq!(quote0.execution_resources.orders_pulled, 1);

        // swap back through the order which should be pulled
        let quote1 = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: to_sqrt_ratio(0),
                override_state: Some(quote0.state_after),
                meta: (),
                token_amount: TokenAmount {
                    token: TOKEN0,
                    amount: 1000,
                },
            })
            .expect("quote1 failed");

        let quote2 = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING),
                override_state: Some(quote1.state_after),
                meta: (),
                token_amount: TokenAmount {
                    token: TOKEN1,
                    amount: 1000,
                },
            })
            .expect("quote2 failed");

        assert_eq!(quote1.consumed_amount, 0);
        assert_eq!(quote1.calculated_amount, 0);
        assert_eq!(quote2.consumed_amount, 0);
        assert_eq!(quote2.calculated_amount, 0);
    }

    #[test]
    fn test_order_sell_token1_for_token0_can_only_be_executed_once() {
        let liquidity: i128 = 10000000;
        let pool = LimitOrderPool::new(
            TOKEN0,
            TOKEN1,
            EXTENSION,
            to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING * 2).unwrap(),
            LIMIT_ORDER_TICK_SPACING * 2,
            liquidity.unsigned_abs(),
            vec![
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING,
                    liquidity_delta: liquidity,
                },
                Tick {
                    index: LIMIT_ORDER_TICK_SPACING * 2,
                    liquidity_delta: -liquidity,
                },
            ],
        );

        // trade all the way through the order
        let quote0 = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING),
                override_state: None,
                meta: (),
                token_amount: TokenAmount {
                    token: TOKEN0,
                    amount: 10000,
                },
            })
            .expect("quote0 failed");

        assert_eq!(
            quote0.state_after.base_pool_state.sqrt_ratio,
            to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING).unwrap()
        );
        assert_eq!(quote0.state_after.base_pool_state.active_tick_index, None);
        assert_eq!(
            quote0.state_after.tick_indices_reached,
            Some((None, Some(1)))
        );

        // swap back through the order which should be pulled
        let quote1 = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING * 2),
                override_state: Some(quote0.state_after),
                meta: (),
                token_amount: TokenAmount {
                    token: TOKEN1,
                    amount: 1000,
                },
            })
            .expect("quote1 failed");

        let quote2 = pool
            .quote(QuoteParams {
                sqrt_ratio_limit: to_sqrt_ratio(LIMIT_ORDER_TICK_SPACING),
                override_state: Some(quote1.state_after),
                meta: (),
                token_amount: TokenAmount {
                    token: TOKEN0,
                    amount: 1000,
                },
            })
            .expect("quote2 failed");

        assert_eq!(quote1.consumed_amount, 0);
        assert_eq!(quote1.calculated_amount, 0);
        assert_eq!(
            quote1.state_after.tick_indices_reached,
            Some((None, Some(1)))
        );
        assert_eq!(quote2.consumed_amount, 0);
        assert_eq!(quote2.calculated_amount, 0);
    }
}
