import pandas as pd
import os
from dotenv import load_dotenv
from utils.sqlite_helper import SQLHelper


load_dotenv()
table_name = os.getenv("data_db_table")

Sqlhelper = SQLHelper()

class BookingAnalytics:
    """
    A class for analyzing hotel booking data.
    
    This class provides methods to analyze various aspects of hotel booking data,
    including cancellation rates, lead times, revenue trends, geographical distribution,
    stay duration, room types, booking channels, pricing, seasonal patterns, deposit types,
    and repeated guest behavior.
    
    Attributes:
        df (pandas.DataFrame): The hotel booking dataset to analyze.
    """
    
    def __init__(self, df=None, csv_path=None):
        """
        Initialize the BookingAnalytics object with a DataFrame.
        
        """
        if df is None:
            self.df = Sqlhelper.data_fetch(table_name=table_name)
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = df

    def cancellation_rate_analysis(self, store=False, update=False):
        """
        Analyze cancellation rates overall, monthly, and yearly.
        
        Calculates the overall cancellation rate as well as monthly and yearly 
        trends. This helps identify time periods with higher cancellation rates
        and track changes over time.
        """
        # Overall cancellation rate
        overall_rate = pd.DataFrame({
            'metric': ['cancellation_rate'],
            'value': [self.df['is_canceled'].mean() * 100]
        })
        
        # Monthly cancellation rate
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        
        monthly_rate = self.df.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
        monthly_rate.columns = ['month', 'cancellation_rate']
        monthly_rate['cancellation_rate'] = monthly_rate['cancellation_rate'] * 100
        
        # Sort by month
        monthly_rate['month_num'] = monthly_rate['month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
        monthly_rate = monthly_rate.sort_values('month_num').drop('month_num', axis=1)
        
        yearly_cancellation = self.df.groupby('arrival_date_year')['is_canceled'].mean().reset_index()
        yearly_cancellation.columns = ["year", "cancellation_rate"]
        yearly_cancellation['cancellation_rate'] = yearly_cancellation['cancellation_rate'] * 100

        if store:
            Sqlhelper.csv_to_sql(overall_rate, 'overall_cancellation_rate', update)
            Sqlhelper.csv_to_sql(monthly_rate, 'monthly_cancellation_rates', update)
            Sqlhelper.csv_to_sql(yearly_cancellation, 'yearly_cancellation_rates', update)

        return {
            'overall': overall_rate,
            'monthly': monthly_rate,
            'yearly': yearly_cancellation
        }
        
    def lead_time_analysis(self, store=False, update=False):
        """
        Analyze booking lead times by cancellation status and distribution.
        
        Lead time is the number of days between booking and actual arrival.
        This analysis helps understand how far in advance bookings are made,
        and whether there's a relationship between lead time and cancellation rates.
        """
        # Lead time statistics by cancellation status
        lead_time_stats = self.df.groupby('is_canceled')['lead_time'].agg([
            'count', 'mean', 'median', 'min', 'max', 'std'
        ]).reset_index()
        
        lead_time_stats['is_canceled'] = lead_time_stats['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'})
        lead_time_stats.rename(columns={'is_canceled': 'booking_status'}, inplace=True)
        
        # Lead time distribution by percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        lead_time_percentiles = self.df['lead_time'].describe(percentiles=[p/100 for p in percentiles])
        lead_time_percentiles = pd.DataFrame(lead_time_percentiles).transpose().reset_index()
        lead_time_percentiles.columns = ['metric'] + ['count', 'mean', 'std', 'min'] + [f'p{p}' for p in percentiles] + ['max']
        
        if store:
            Sqlhelper.csv_to_sql(lead_time_stats, 'lead_time_by_cancellation', update)
            Sqlhelper.csv_to_sql(lead_time_percentiles, 'lead_time_percentiles', update)
            
        return {
            'by_cancellation': lead_time_stats,
            'percentiles': lead_time_percentiles
        }
    
    def revenue_trend_analysis(self, store=False, update=False):
        """
        Analyze revenue trends by time period and country.
        
        Calculates revenue based on Average Daily Rate (ADR) and length of stay.
        This helps identify peak revenue periods, trends over time, and
        most valuable source markets.
        """
        # Create revenue column if not exists
        if 'revenue' not in self.df.columns:
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            self.df['revenue'] = self.df['adr'] * self.df['total_nights']
        
        # Revenue by year and month (excluding canceled bookings)
        revenue_trend = self.df[self.df['is_canceled'] == 0].groupby(
            ['arrival_date_year', 'arrival_date_month']
        )['revenue'].agg(['sum', 'mean', 'count']).reset_index()
        
        # Calculate month_num for proper sorting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        revenue_trend['month_num'] = revenue_trend['arrival_date_month'].apply(
            lambda x: month_order.index(x) if x in month_order else -1
        )
        
        # Sort by year and month
        revenue_trend = revenue_trend.sort_values(['arrival_date_year', 'month_num'])
        
        # Format columns for clarity
        revenue_trend.rename(columns={
            'sum': 'total_revenue',
            'mean': 'avg_revenue_per_booking',
            'count': 'booking_count'
        }, inplace=True)
        
        # Drop the helper column
        revenue_trend = revenue_trend.drop('month_num', axis=1)
        
        # Calculate country-wise revenue
        country_wise_revenue = self.df[self.df['is_canceled'] == 0].groupby('country')['revenue'].sum().reset_index()
        country_wise_revenue = country_wise_revenue.sort_values('revenue', ascending=False)

        if store:
            Sqlhelper.csv_to_sql(revenue_trend, 'monthly_revenue_trend', update)
            Sqlhelper.csv_to_sql(country_wise_revenue, 'country_wise_revenue', update)
            
        return {
            'revenue_trend': revenue_trend,
            'country_wise_revenue': country_wise_revenue
        }

    def geographical_distribution(self, store=False, update=False):
        """
        Analyze the geographical distribution of bookings by country.
        
        Identifies the most important source markets for the hotel,
        including booking volumes and cancellation rates by country.
        """
        # Country distribution
        country_dist = self.df['country'].value_counts().reset_index()
        country_dist.columns = ['country', 'booking_count']
        
        # Add percentage
        country_dist['percentage'] = (country_dist['booking_count'] / len(self.df) * 100).round(2)
        
        # Add cancellation rate by country
        country_cancel = self.df.groupby('country')['is_canceled'].mean().reset_index()
        country_cancel.columns = ['country', 'cancellation_rate']
        country_cancel['cancellation_rate'] = (country_cancel['cancellation_rate'] * 100).round(2)
        
        # Merge the two dataframes
        country_analysis = pd.merge(country_dist, country_cancel, on='country', how='left')
        
        # Sort by booking count in descending order
        country_analysis = country_analysis.sort_values('booking_count', ascending=False)
        
        if store:
            Sqlhelper.csv_to_sql(country_analysis, 'country_distribution', update)
            
        return country_analysis

    def stay_duration_analysis(self, store=False, update=False):
        """
        Analyze guest stay duration patterns and distribution.
        
        Examines how long guests typically stay, whether there are differences
        in stay duration between canceled and non-canceled bookings, and
        identifies the most common length of stay.
        """
        # Create stay_duration if not exists
        if 'stay_duration' not in self.df.columns:
            self.df['stay_duration'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        
        # Overall statistics
        stay_stats = self.df.groupby('is_canceled')['stay_duration'].agg([
            'count', 'mean', 'median', 'min', 'max', 'std'
        ]).reset_index()
        
        stay_stats['is_canceled'] = stay_stats['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'})
        stay_stats.rename(columns={'is_canceled': 'booking_status'}, inplace=True)
        
        # Stay duration frequency
        stay_freq = self.df[self.df['is_canceled'] == 0]['stay_duration'].value_counts().reset_index()
        stay_freq.columns = ['stay_nights', 'frequency']
        stay_freq = stay_freq.sort_values('stay_nights')
        
        if store:
            Sqlhelper.csv_to_sql(stay_stats, 'stay_duration_stats', update)
            Sqlhelper.csv_to_sql(stay_freq, 'stay_duration_frequency', update)
            
        return {
            'stats': stay_stats,
            'frequency': stay_freq
        }

    def room_type_analysis(self, store=False, update=False):
        """
        Analyze room type distribution and room assignment changes.
        
        Examines the distribution of reserved vs. assigned room types and
        calculates the room change rate (when guests are assigned a different
        room than what they reserved). This helps understand room type 
        popularity and operational efficiency.
        """
        # Reserved room type distribution
        reserved_rooms = self.df['reserved_room_type'].value_counts().reset_index()
        reserved_rooms.columns = ['room_type', 'reserved_count']
        reserved_rooms['reserved_percentage'] = (reserved_rooms['reserved_count'] / len(self.df) * 100).round(2)
        
        # Assigned room type distribution
        assigned_rooms = self.df['assigned_room_type'].value_counts().reset_index()
        assigned_rooms.columns = ['room_type', 'assigned_count']
        
        # Merge the two
        room_analysis = pd.merge(reserved_rooms, assigned_rooms, on='room_type', how='outer').fillna(0)
        
        # Calculate room change rate
        room_change = self.df[self.df['reserved_room_type'] != self.df['assigned_room_type']].shape[0]
        room_change_rate = pd.DataFrame({
            'metric': ['room_change_rate'],
            'value': [room_change / len(self.df) * 100]
        })
        
        if store:
            Sqlhelper.csv_to_sql(room_analysis, 'room_type_distribution', update)
            Sqlhelper.csv_to_sql(room_change_rate, 'room_change_rate', update)
            
        return {
            'distribution': room_analysis,
            'change_rate': room_change_rate
        }
    
    def booking_channel_analysis(self, store=False, update=False):
        """
        Analyze distribution channels and market segments for bookings.
        
        Identifies the most important booking channels and market segments,
        including their volume and cancellation rates. This helps optimize
        marketing and distribution strategies.
        """
        # Distribution channel analysis
        channel_dist = self.df['distribution_channel'].value_counts().reset_index()
        channel_dist.columns = ['distribution_channel', 'booking_count']
        channel_dist['percentage'] = (channel_dist['booking_count'] / len(self.df) * 100).round(2)
        
        # Market segment analysis
        segment_dist = self.df['market_segment'].value_counts().reset_index()
        segment_dist.columns = ['market_segment', 'booking_count']
        segment_dist['percentage'] = (segment_dist['booking_count'] / len(self.df) * 100).round(2)
        
        # Cancellation rates by channel
        channel_cancel = self.df.groupby('distribution_channel')['is_canceled'].mean().reset_index()
        channel_cancel.columns = ['distribution_channel', 'cancellation_rate']
        channel_cancel['cancellation_rate'] = (channel_cancel['cancellation_rate'] * 100).round(2)
        
        # Merge channel distribution and cancellation
        channel_analysis = pd.merge(channel_dist, channel_cancel, on='distribution_channel')
        
        if store:
            Sqlhelper.csv_to_sql(channel_analysis, 'distribution_channel_analysis', update)
            Sqlhelper.csv_to_sql(segment_dist, 'market_segment_analysis', update)
            
        return {
            'channels': channel_analysis,
            'segments': segment_dist
        }

    def price_analysis(self, store=False, update=False):
        """
        Analyze Average Daily Rate (ADR) patterns and distribution.
        
        Examines price variations by room type, market segment, and
        season. This helps understand pricing dynamics and identify
        opportunities for revenue optimization.
        """
        # Overall price statistics
        price_stats = self.df['adr'].describe().reset_index()
        price_stats.columns = ['metric', 'value']
        
        # Price by room type
        price_by_room = self.df.groupby('reserved_room_type')['adr'].agg([
            'count', 'mean', 'median', 'min', 'max'
        ]).reset_index()
        
        # Price by market segment
        price_by_segment = self.df.groupby('market_segment')['adr'].agg([
            'count', 'mean', 'median', 'min', 'max'
        ]).reset_index()
        
        # Monthly price trends
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
        
        monthly_price = self.df.groupby('arrival_date_month')['adr'].mean().reset_index()
        monthly_price.columns = ['month', 'average_price']
        
        # Sort by month
        monthly_price['month_num'] = monthly_price['month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
        monthly_price = monthly_price.sort_values('month_num').drop('month_num', axis=1)
        
        if store:
            Sqlhelper.csv_to_sql(price_stats, 'price_statistics', update)
            Sqlhelper.csv_to_sql(price_by_room, 'price_by_room_type', update)
            Sqlhelper.csv_to_sql(price_by_segment, 'price_by_market_segment', update)
            Sqlhelper.csv_to_sql(monthly_price, 'monthly_price_trend', update)
            
        return {
            'stats': price_stats,
            'by_room': price_by_room,
            'by_segment': price_by_segment,
            'monthly': monthly_price
        }
        
    def seasonal_patterns(self, store=False, update=False):
        """
        Analyze seasonal booking and cancellation patterns.
        
        Identifies peak booking seasons, high and low demand periods,
        and seasonal variations in cancellation rates. This helps with
        capacity planning and seasonal marketing strategies.
        """
        # Bookings by month
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        
        bookings_by_month = self.df.groupby('arrival_date_month').size().reset_index()
        bookings_by_month.columns = ['month', 'booking_count']
        
        # Sort by month
        bookings_by_month['month_num'] = bookings_by_month['month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
        bookings_by_month = bookings_by_month.sort_values('month_num').drop('month_num', axis=1)
        
        # Add confirmed bookings
        confirmed_bookings = self.df[self.df['is_canceled'] == 0].groupby('arrival_date_month').size().reset_index()
        confirmed_bookings.columns = ['month', 'confirmed_count']
        
        # Merge
        monthly_patterns = pd.merge(bookings_by_month, confirmed_bookings, on='month', how='left')
        monthly_patterns['cancellation_count'] = monthly_patterns['booking_count'] - monthly_patterns['confirmed_count']
        monthly_patterns['cancellation_rate'] = (monthly_patterns['cancellation_count'] / monthly_patterns['booking_count'] * 100).round(2)
        
        # Sort by month
        monthly_patterns['month_num'] = monthly_patterns['month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
        monthly_patterns = monthly_patterns.sort_values('month_num').drop('month_num', axis=1)
        
        if store:
            Sqlhelper.csv_to_sql(monthly_patterns, 'seasonal_booking_patterns', update)
            
        return monthly_patterns

    def deposit_type_analysis(self, store=False, update=False):
        """
        Analyze impact of deposit types on cancellation rates and revenue.
        
        Examines how different deposit policies (no deposit, non-refundable, refundable)
        affect cancellation behavior and revenue. This helps optimize 
        deposit policies to balance bookings and revenue.
        """
        # Distribution of deposit types
        deposit_dist = self.df['deposit_type'].value_counts().reset_index()
        deposit_dist.columns = ['deposit_type', 'booking_count']
        deposit_dist['percentage'] = (deposit_dist['booking_count'] / len(self.df) * 100).round(2)
        
        # Cancellation rate by deposit type
        deposit_cancel = self.df.groupby('deposit_type')['is_canceled'].mean().reset_index()
        deposit_cancel.columns = ['deposit_type', 'cancellation_rate']
        deposit_cancel['cancellation_rate'] = (deposit_cancel['cancellation_rate'] * 100).round(2)
        
        # Merge
        deposit_analysis = pd.merge(deposit_dist, deposit_cancel, on='deposit_type')
        
        # Add revenue analysis if revenue column exists
        if 'revenue' in self.df.columns:
            deposit_revenue = self.df[self.df['is_canceled'] == 0].groupby('deposit_type')['revenue'].agg([
                'sum', 'mean', 'count'
            ]).reset_index()
            
            deposit_revenue.rename(columns={
                'sum': 'total_revenue',
                'mean': 'avg_revenue_per_booking',
                'count': 'confirmed_bookings'
            }, inplace=True)
            
            deposit_analysis = pd.merge(deposit_analysis, deposit_revenue, on='deposit_type', how='left')
        
        if store:
            Sqlhelper.csv_to_sql(deposit_analysis, 'deposit_type_analysis', update)
            
        return deposit_analysis
    
    def repeated_guest_analysis(self, store=False, update=False):
        """
        Analyze behavior patterns of repeat guests versus new guests.
        
        Examines differences in booking and cancellation behavior between
        repeat and new guests. This helps understand customer loyalty and
        develop targeted retention strategies.
        """
        # Distribution of repeated guests
        repeated = self.df['is_repeated_guest'].value_counts().reset_index()
        repeated.columns = ['is_repeated_guest', 'booking_count']
        repeated['percentage'] = (repeated['booking_count'] / len(self.df) * 100).round(2)
        
        # Map binary values to labels
        repeated['is_repeated_guest'] = repeated['is_repeated_guest'].map({0: 'New Guest', 1: 'Repeat Guest'})
        
        # Cancellation behavior
        repeat_cancel = self.df.groupby('is_repeated_guest')['is_canceled'].mean().reset_index()
        repeat_cancel.columns = ['is_repeated_guest', 'cancellation_rate']
        repeat_cancel['cancellation_rate'] = (repeat_cancel['cancellation_rate'] * 100).round(2)
        repeat_cancel['is_repeated_guest'] = repeat_cancel['is_repeated_guest'].map({0: 'New Guest', 1: 'Repeat Guest'})
        
        # Merge
        repeat_analysis = pd.merge(repeated, repeat_cancel, on='is_repeated_guest')
        
        if store:
            Sqlhelper.csv_to_sql(repeat_analysis, 'repeat_guest_analysis', update)
            
        return repeat_analysis

    def high_cancellation_factors(self, store=False, update=False):
        """
        Identify factors associated with high cancellation rates.
        
        Analyzes various booking attributes to identify factors that are
        strongly associated with cancellations. This helps develop targeted
        strategies to reduce cancellations for high-risk bookings.
        """
        # Analyze categorical variables for cancellation patterns
        cat_vars = ['hotel', 'meal', 'market_segment', 'distribution_channel', 
                    'deposit_type', 'customer_type', 'reserved_room_type']
        
        cancellation_factors = []
        
        for var in cat_vars:
            if var in self.df.columns:
                # Calculate cancellation rate by factor
                factor_cancel = self.df.groupby(var)['is_canceled'].mean().reset_index()
                factor_cancel.columns = [var, 'cancellation_rate']
                factor_cancel['cancellation_rate'] = (factor_cancel['cancellation_rate'] * 100).round(2)
                
                # Add booking count
                factor_count = self.df.groupby(var).size().reset_index()
                factor_count.columns = [var, 'booking_count']
                
                # Merge
                factor_analysis = pd.merge(factor_cancel, factor_count, on=var)
                factor_analysis['factor_type'] = var
                factor_analysis.rename(columns={var: 'factor_value'}, inplace=True)
                
                # Reorder columns
                factor_analysis = factor_analysis[['factor_type', 'factor_value', 'cancellation_rate', 'booking_count']]
                
                cancellation_factors.append(factor_analysis)
        
        # Combine all factors
        if cancellation_factors:
            all_factors = pd.concat(cancellation_factors)
            all_factors = all_factors.sort_values('cancellation_rate', ascending=False)
            
            if store:
                Sqlhelper.csv_to_sql(all_factors, 'high_cancellation_factors', update)
                
            return all_factors
        else:
            return pd.DataFrame()
        
    def create_all_analytics(self, store=True, update=True):
        """
        Generate all analytics tables and optionally store them in the database.
        
        This is a convenience method that runs all analytics methods and
        compiles their results into a single dictionary. Useful for generating
        a comprehensive analysis in one step.
        
        """
        try:
            results = {}
            
            print("Checking for existing tables...")
            all_tables = [
                'overall_cancellation_rate', 'monthly_cancellation_rates', 'yearly_cancellation_rates',
                'lead_time_by_cancellation', 'lead_time_percentiles',
                'monthly_revenue_trend', 'country_wise_revenue',
                'country_distribution',
                'stay_duration_stats', 'stay_duration_frequency',
                'room_type_distribution', 'room_change_rate',
                'distribution_channel_analysis', 'market_segment_analysis',
                'price_statistics', 'price_by_room_type', 'price_by_market_segment', 'monthly_price_trend',
                'seasonal_booking_patterns',
                'deposit_type_analysis',
                'repeat_guest_analysis',
                'high_cancellation_factors'
            ]
            existing_tables = Sqlhelper.check_table_exists(all_tables)
            
            analytics_functions = {
                'cancellation_rates': {
                    'function': self.cancellation_rate_analysis,
                    'tables': ['overall_cancellation_rate', 'monthly_cancellation_rates', 'yearly_cancellation_rates']
                },
                'lead_time': {
                    'function': self.lead_time_analysis,
                    'tables': ['lead_time_by_cancellation', 'lead_time_percentiles']
                },
                'revenue_trends': {
                    'function': self.revenue_trend_analysis,
                    'tables': ['monthly_revenue_trend', 'country_wise_revenue']
                },
                'geographical_distribution': {
                    'function': self.geographical_distribution,
                    'tables': ['country_distribution']
                },
                'stay_duration': {
                    'function': self.stay_duration_analysis,
                    'tables': ['stay_duration_stats', 'stay_duration_frequency']
                },
                'room_types': {
                    'function': self.room_type_analysis,
                    'tables': ['room_type_distribution', 'room_change_rate']
                },
                'booking_channels': {
                    'function': self.booking_channel_analysis,
                    'tables': ['distribution_channel_analysis', 'market_segment_analysis']
                },
                'prices': {
                    'function': self.price_analysis,
                    'tables': ['price_statistics', 'price_by_room_type', 'price_by_market_segment', 'monthly_price_trend']
                },
                'seasonal_patterns': {
                    'function': self.seasonal_patterns,
                    'tables': ['seasonal_booking_patterns']
                },
                'deposit_types': {
                    'function': self.deposit_type_analysis,
                    'tables': ['deposit_type_analysis']
                },
                'repeated_guests': {
                    'function': self.repeated_guest_analysis,
                    'tables': ['repeat_guest_analysis']
                },
                'cancellation_factors': {
                    'function': self.high_cancellation_factors,
                    'tables': ['high_cancellation_factors']
                }
            }
            
            for key, info in analytics_functions.items():
                should_run = False
                
                if update:
                    should_run = True
                else:
                    for table in info['tables']:
                        if not existing_tables.get(table, False):
                            should_run = True
                            break
                
                if should_run:
                    print(f"Generating {key}...")
                    result = info['function'](store, update)
                    if result is not None:
                        results[key] = result
                else:
                    print(f"Skipping {key} - tables exist and update=False")
            
            if results:
                print("All requested analytics generated successfully!")
            else:
                print("No new analytics were generated. All tables exist and update=False")
                
            return results
        except Exception as e:
            print(f"Error generating analytics: {e}")
            return {}