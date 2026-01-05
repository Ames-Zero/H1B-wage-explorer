"""
H1B Wage Interactive Dashboard
Visualizes H1B wage data across US geographical regions with filtering capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from addfips import AddFIPS
import numpy as np

# Page configuration
st.set_page_config(
    page_title="H1B Wage Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize FIPS converter
af = AddFIPS()

@st.cache_data
def load_data():
    """Load and merge all data files with FIPS code generation"""
    
    # Load geographic data
    geography = pd.read_csv('OFLC_Wages_2025-26_Updated/Geography.csv')
    
    # Load job role mapping
    job_mapping = pd.read_csv('OFLC_Wages_2025-26_Updated/oes_soc_occs.csv')
    
    # Load wage data - we'll use ALC_Export as the primary source
    # EDC_Export has similar structure but may have different coverage
    wage_data_alc = pd.read_csv('OFLC_Wages_2025-26_Updated/ALC_Export.csv', low_memory=False)
    wage_data_edc = pd.read_csv('OFLC_Wages_2025-26_Updated/EDC_Export.csv', low_memory=False)
    
    # Combine wage data (prioritize ALC, fill with EDC where missing)
    wage_data = pd.concat([wage_data_alc, wage_data_edc], ignore_index=True)
    wage_data = wage_data.drop_duplicates(subset=['Area', 'SocCode'], keep='first')
    
    # Convert Area to string and ensure consistent format
    geography['Area'] = geography['Area'].astype(str)
    wage_data['Area'] = wage_data['Area'].astype(str)
    
    # Convert wage columns to numeric first
    wage_columns = ['Level1', 'Level2', 'Level3', 'Level4']
    for col in wage_columns:
        wage_data[col] = pd.to_numeric(wage_data[col], errors='coerce')
    
    # Clean the data: convert annual wages to hourly
    # Label column contains: "" (blank = hourly), "Annual Wage", or "High Wage"
    wage_data['Label'] = wage_data['Label'].fillna('')
    
    # Convert annual wages to hourly by dividing by 2080 (40 hours/week * 52 weeks)
    annual_mask = wage_data['Label'].str.contains('Annual', case=False, na=False)
    for col in wage_columns:
        wage_data.loc[annual_mask, col] = wage_data.loc[annual_mask, col] / 2080
    
    # Generate FIPS codes for each county
    geography['fips'] = geography.apply(
        lambda row: af.get_county_fips(row['CountyTownName'], state=row['StateAb']),
        axis=1
    )
    
    # Merge geography with wage data
    merged_data = geography.merge(
        wage_data,
        on='Area',
        how='inner'
    )
    
    # Merge with job mapping
    merged_data = merged_data.merge(
        job_mapping,
        left_on='SocCode',
        right_on='soccode',
        how='left'
    )
    
    # Rename columns for clarity
    merged_data = merged_data.rename(columns={
        'Title': 'job_role',
        'Level1': 'wage_level_1',
        'Level2': 'wage_level_2',
        'Level3': 'wage_level_3',
        'Level4': 'wage_level_4'
    })
    
    # Handle missing FIPS codes
    merged_data = merged_data.dropna(subset=['fips'])
    
    # Ensure FIPS is 5-digit string
    merged_data['fips'] = merged_data['fips'].astype(str).str.zfill(5)
    
    return merged_data

@st.cache_data
def aggregate_by_state(data, wage_column):
    """Aggregate wage data by state"""
    state_data = data.groupby('StateAb')[wage_column].mean().reset_index()
    state_data.columns = ['StateAb', 'avg_wage']
    return state_data

@st.cache_data
def calculate_statistics(data, wage_column):
    """Calculate wage statistics"""
    valid_wages = data[wage_column].dropna()
    
    if len(valid_wages) == 0:
        return {
            'avg': 0,
            'median': 0,
            'max': 0,
            'min': 0
        }
    
    return {
        'avg': valid_wages.mean(),
        'median': valid_wages.median(),
        'max': valid_wages.max(),
        'min': valid_wages.min()
    }

def format_currency(value):
    """Format value as hourly wage"""
    return f"${value:,.2f}/hr"

def format_annual_currency(hourly_value):
    """Convert hourly wage to annual and format"""
    annual = hourly_value * 2080  # 40 hours/week * 52 weeks
    return f"${annual:,.0f}"

def format_combined_wage(hourly_value):
    """Format as both hourly and annual wage"""
    annual = hourly_value * 2080  # 40 hours/week * 52 weeks
    return f"${hourly_value:,.2f}/hr (${annual:,.0f}/yr)"

def classify_wage_level(row, hourly_salary):
    """Determine which wage level the salary falls into"""
    level1 = row.get('wage_level_1', 0) or 0
    level2 = row.get('wage_level_2', 0) or 0
    level3 = row.get('wage_level_3', 0) or 0
    level4 = row.get('wage_level_4', 0) or 0
    
    if pd.isna(hourly_salary) or hourly_salary <= 0:
        return 'Unknown'
    
    if hourly_salary < level1:
        return 'Below Level 1'
    elif hourly_salary < level2:
        return 'Level 1 (Entry)'
    elif hourly_salary < level3:
        return 'Level 2 (Qualified)'
    elif hourly_salary < level4:
        return 'Level 3 (Experienced)'
    else:
        return 'Level 4+ (Fully Competent)'

# Main app
def main():
    # Title and description
    st.title("🌎 H1B Wage Explorer")
    st.markdown("""
    Explore H1B prevailing wage data across the United States. Filter by job role, 
    wage level, and view data at state or county level.
    
    **Note:** Wage levels represent different experience levels (1=Entry, 2=Qualified, 3=Experienced, 4=Fully Competent)
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Visualization mode
    viz_mode = st.sidebar.radio(
        "Visualization Mode",
        options=['Wage Amount', 'Salary Classification'],
        index=0,
        help="Wage Amount: View wage levels. Salary Classification: See which level your salary falls into."
    )
    
    # Job role filter
    job_roles = ['All'] + sorted(data['job_role'].dropna().unique().tolist())
    selected_role = st.sidebar.selectbox(
        "Select Job Role",
        options=job_roles,
        index=0
    )
    
    # Conditional filters based on mode
    if viz_mode == 'Wage Amount':
        # Wage level filter
        wage_level = st.sidebar.radio(
            "Wage Level",
            options=['Level 1', 'Level 2', 'Level 3', 'Level 4'],
            index=1,
            help="Level 1: Entry, Level 2: Qualified, Level 3: Experienced, Level 4: Fully Competent"
        )
        user_salary = None
    else:
        # Salary input for classification
        st.sidebar.markdown("---")
        salary_input_type = st.sidebar.radio(
            "Salary Input Type",
            options=['Annual', 'Hourly'],
            index=0
        )
        
        if salary_input_type == 'Annual':
            annual_salary = st.sidebar.number_input(
                "Your Annual Salary ($)",
                min_value=0,
                max_value=500000,
                value=80000,
                step=5000,
                help="Enter your current annual salary"
            )
            user_salary = annual_salary / 2080  # Convert to hourly
        else:
            user_salary = st.sidebar.number_input(
                "Your Hourly Wage ($)",
                min_value=0.0,
                max_value=250.0,
                value=38.46,
                step=1.0,
                help="Enter your current hourly wage"
            )
        
        wage_level = None
        st.sidebar.info(f"Hourly equivalent: ${user_salary:.2f}/hr")
    
    # Map detail filter
    map_detail = st.sidebar.radio(
        "Map Detail",
        options=['State-Level (faster)', 'County-Level (detailed)'],
        index=0
    )
    
    # Filter data based on selections
    filtered_data = data.copy()
    
    if selected_role != 'All':
        filtered_data = filtered_data[filtered_data['job_role'] == selected_role]
    
    # Determine wage column or add classification
    if viz_mode == 'Wage Amount':
        wage_column = f"wage_level_{wage_level.split()[-1]}"
        # Remove rows with null wages for the selected level
        filtered_data = filtered_data.dropna(subset=[wage_column])
    else:
        # Add wage level classification
        filtered_data['wage_classification'] = filtered_data.apply(
            lambda row: classify_wage_level(row, user_salary),
            axis=1
        )
        wage_column = 'wage_classification'
        # Remove rows where classification failed
        filtered_data = filtered_data[filtered_data[wage_column] != 'Unknown']
    
    if len(filtered_data) == 0:
        st.warning("No data available for the selected filters. Please try different options.")
        return
    
    # Check data size for county-level rendering
    if map_detail == 'County-Level (detailed)' and selected_role == 'All' and len(filtered_data) > 100000:
        st.warning("⚠️ County-level view with 'All' job roles contains too much data. Please select a specific job role for county-level visualization, or use State-Level view.")
        return
    
    # Create visualization
    st.subheader("📊 Wage Distribution Map")
    
    if viz_mode == 'Wage Amount':
        if map_detail == 'State-Level (faster)':
            # Aggregate by state
            plot_data = aggregate_by_state(filtered_data, wage_column)
            # Add full state names
            state_names = filtered_data[['StateAb', 'State']].drop_duplicates()
            plot_data = plot_data.merge(state_names, on='StateAb', how='left')
            
            fig = px.choropleth(
                plot_data,
                locations='StateAb',
                locationmode='USA-states',
                color='avg_wage',
                scope='usa',
                color_continuous_scale='Viridis',
                labels={'avg_wage': 'Avg Hourly Wage ($)'},
                title=f'{selected_role} - {wage_level} Wages by State',
                height=600
            )
            
            fig.update_traces(
                hovertemplate='<b>%{customdata[0]}</b><br>Hourly: $%{z:.2f}/hr<br>Annual: $%{customdata[1]:,.0f}/yr<extra></extra>',
                customdata=plot_data[['State', 'avg_wage']].apply(lambda x: [x['State'], x['avg_wage'] * 2080], axis=1).tolist()
            )
        
    else:  # Salary Classification Mode
        if map_detail == 'State-Level (faster)':
            # Aggregate classifications by state - use mode (most common)
            state_classification = filtered_data.groupby('StateAb')['wage_classification'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            ).reset_index()
            state_classification.columns = ['StateAb', 'classification']
            # Add full state names
            state_names = filtered_data[['StateAb', 'State']].drop_duplicates()
            state_classification = state_classification.merge(state_names, on='StateAb', how='left')
            
            # Define color mapping for wage levels
            color_map = {
                'Below Level 1': '#d62728',  # Red
                'Level 1 (Entry)': '#ff7f0e',  # Orange
                'Level 2 (Qualified)': '#2ca02c',  # Green
                'Level 3 (Experienced)': '#1f77b4',  # Blue
                'Level 4+ (Fully Competent)': '#9467bd'  # Purple
            }
            
            fig = px.choropleth(
                state_classification,
                locations='StateAb',
                locationmode='USA-states',
                color='classification',
                scope='usa',
                color_discrete_map=color_map,
                category_orders={'classification': ['Below Level 1', 'Level 1 (Entry)', 'Level 2 (Qualified)', 
                                                     'Level 3 (Experienced)', 'Level 4+ (Fully Competent)']},
                labels={'classification': 'Wage Level Classification'},
                title=f'{selected_role} - Your Salary Classification by State (${user_salary:.2f}/hr)',
                height=600
            )
            
            fig.update_traces(
                hovertemplate='<b>%{customdata}</b><br>Classification: %{z}<extra></extra>',
                customdata=state_classification['State']
            )
        
    if viz_mode == 'Wage Amount' and map_detail == 'County-Level (detailed)':
        # Simplify data to only essential columns to reduce payload size
        plot_data = filtered_data[['fips', 'CountyTownName', 'State', 'AreaName', wage_column]].copy()
        plot_data['annual_wage'] = plot_data[wage_column] * 2080
        
        fig = px.choropleth(
            plot_data,
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            locations='fips',
            color=wage_column,
            scope='usa',
            color_continuous_scale='Viridis',
            labels={wage_column: 'Hourly Wage ($)'},
            title=f'{selected_role} - {wage_level} Wages by County',
            height=600
        )
        
        fig.update_traces(
            hovertemplate='<b>%{customdata[0]}, %{customdata[1]}</b><br>' +
                         'Area: %{customdata[2]}<br>' +
                         'Hourly: $%{z:.2f}/hr<br>' +
                         'Annual: $%{customdata[3]:,.0f}/yr<extra></extra>',
            customdata=plot_data[['CountyTownName', 'State', 'AreaName', 'annual_wage']].values
        )
    
    elif viz_mode == 'Salary Classification' and map_detail == 'County-Level (detailed)':
        # Simplify data to only essential columns
        plot_data = filtered_data[['fips', 'CountyTownName', 'State', 'AreaName', 'wage_classification']].copy()
        
        # Define color mapping for wage levels
        color_map = {
            'Below Level 1': '#d62728',  # Red
            'Level 1 (Entry)': '#ff7f0e',  # Orange
            'Level 2 (Qualified)': '#2ca02c',  # Green
            'Level 3 (Experienced)': '#1f77b4',  # Blue
            'Level 4+ (Fully Competent)': '#9467bd'  # Purple
        }
        
        fig = px.choropleth(
            plot_data,
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            locations='fips',
            color='wage_classification',
            scope='usa',
            color_discrete_map=color_map,
            category_orders={'wage_classification': ['Below Level 1', 'Level 1 (Entry)', 'Level 2 (Qualified)', 
                                                       'Level 3 (Experienced)', 'Level 4+ (Fully Competent)']},
            labels={'wage_classification': 'Wage Level Classification'},
            title=f'{selected_role} - Your Salary Classification by County (${user_salary:.2f}/hr)',
            height=600
        )
        
        fig.update_traces(
            hovertemplate='<b>%{customdata[0]}, %{customdata[1]}</b><br>' +
                         'Area: %{customdata[2]}<br>' +
                         'Classification: %{z}<extra></extra>',
            customdata=plot_data[['CountyTownName', 'State', 'AreaName']].values
        )
    
    fig.update_layout(
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='lightblue'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    if viz_mode == 'Wage Amount':
        st.subheader("📈 Wage Statistics")
        stats = calculate_statistics(filtered_data, wage_column)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Average Wage",
                value=format_currency(stats['avg']),
                delta=format_annual_currency(stats['avg']) + " annually"
            )
        
        with col2:
            st.metric(
                label="Median Wage",
                value=format_currency(stats['median']),
                delta=format_annual_currency(stats['median']) + " annually"
            )
        
        with col3:
            st.metric(
                label="Maximum Wage",
                value=format_currency(stats['max']),
                delta=format_annual_currency(stats['max']) + " annually"
            )
        
        with col4:
            st.metric(
                label="Minimum Wage",
                value=format_currency(stats['min']),
                delta=format_annual_currency(stats['min']) + " annually"
            )
    else:
        st.subheader("📈 Salary Classification Distribution")
        classification_counts = filtered_data['wage_classification'].value_counts()
        total_areas = len(filtered_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            level1_count = classification_counts.get('Level 1 (Entry)', 0)
            st.metric(
                label="Level 1 (Entry)",
                value=f"{level1_count:,} areas",
                delta=f"{(level1_count/total_areas*100):.1f}% of areas"
            )
        
        with col2:
            level2_count = classification_counts.get('Level 2 (Qualified)', 0)
            st.metric(
                label="Level 2 (Qualified)",
                value=f"{level2_count:,} areas",
                delta=f"{(level2_count/total_areas*100):.1f}% of areas"
            )
        
        with col3:
            level3_count = classification_counts.get('Level 3 (Experienced)', 0)
            st.metric(
                label="Level 3 (Experienced)",
                value=f"{level3_count:,} areas",
                delta=f"{(level3_count/total_areas*100):.1f}% of areas"
            )
        
        with col4:
            level4_count = classification_counts.get('Level 4+ (Fully Competent)', 0)
            st.metric(
                label="Level 4+ (Fully Competent)",
                value=f"{level4_count:,} areas",
                delta=f"{(level4_count/total_areas*100):.1f}% of areas"
            )
    
    # Data table
    with st.expander("🔍 View Raw Data (First 100 Rows)"):
        if viz_mode == 'Wage Amount':
            display_columns = [
                'State', 'AreaName', 'CountyTownName', 
                'job_role', wage_column
            ]
            
            display_data = filtered_data[display_columns].head(100).copy()
            display_data['Hourly Wage'] = display_data[wage_column].apply(
                lambda x: format_currency(x)
            )
            display_data['Annual Wage'] = display_data[wage_column].apply(
                lambda x: format_annual_currency(x)
            )
            display_data = display_data.drop(columns=[wage_column])
        else:
            display_columns = [
                'State', 'AreaName', 'CountyTownName', 
                'job_role', 'wage_classification', 'wage_level_1', 'wage_level_2', 'wage_level_3', 'wage_level_4'
            ]
            
            display_data = filtered_data[display_columns].head(100).copy()
            display_data['Your Classification'] = display_data['wage_classification']
            display_data['Level 1 Range'] = display_data['wage_level_1'].apply(
                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
            )
            display_data['Level 2 Range'] = display_data['wage_level_2'].apply(
                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
            )
            display_data['Level 3 Range'] = display_data['wage_level_3'].apply(
                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
            )
            display_data['Level 4 Range'] = display_data['wage_level_4'].apply(
                lambda x: format_currency(x) if pd.notna(x) else 'N/A'
            )
            display_data = display_data.drop(columns=['wage_classification', 'wage_level_1', 'wage_level_2', 'wage_level_3', 'wage_level_4'])
        
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Data Points:** {len(filtered_data):,} records")
    st.sidebar.info(f"**Unique Areas:** {filtered_data['Area'].nunique():,}")

if __name__ == "__main__":
    main()
