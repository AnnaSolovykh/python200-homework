import pandas as pd
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

@task(retries=3, retry_delay_seconds=2)
def load_merge_data():
    logger=get_run_logger()
    logger.info("Loading data")

    all_data = []

    for year in range (2015, 2025):
        file_path = f"resources/happiness_project/world_happiness_{year}.csv"
        df_year = pd.read_csv(file_path, sep=";", decimal=",")
        
        # Fixing the quirks: We assume "Happiness score" (used in older files like 2015) 
        # and "Ladder score" (used in newer files) represent the exact same metric 
        # (the Cantril Ladder survey). We rename both to a single "happiness_score" 
        # so Pandas merges them into one column without creating NaNs.
        df_year = df_year.rename(columns={
            "Ladder score": "happiness_score", 
            "Happiness score": "happiness_score"
        })
        
        df_year['Year'] = year
        all_data.append(df_year)
    
    merged_df = pd.concat(all_data, ignore_index=True)
    output_path = 'assignments_01/outputs/merged_happiness.csv'
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f'Successfully saved merged data')
    
    return merged_df

@task
def descriptive_statistics(df):
    logger = get_run_logger()
    
    happiness = df["happiness_score"]
    logger.info(f"Happiness -> Mean: {happiness.mean():.3f}, Median: {happiness.median():.3f}, STD: {happiness.std():.3f}")
    
    logger.info(f"Mean Happiness by Year:\n{df.groupby('Year')['happiness_score'].mean()}")
    logger.info(f"Mean Happiness by Region:\n{df.groupby('Regional indicator')['happiness_score'].mean().sort_values(ascending=False)}")

@task
def visualize_data(df):
    logger=get_run_logger()

    plt.clf()
    sns.histplot(df['happiness_score'], bins=20)
    plt.title('Distribution of Happiness Scores')
    plt.savefig('assignments_01/outputs/happiness_histogram.png')
    logger.info('Saved happiness histogram')

    plt.clf()
    sns.boxplot(data=df, x="Year", y="happiness_score")
    plt.title("Happiness by Year")
    plt.savefig("assignments_01/outputs/happiness_by_year.png")
    logger.info("Saved happiness_by_year.png")

    plt.clf()
    sns.scatterplot(data=df, x="GDP per capita", y="happiness_score")
    plt.title("GDP per Capita vs Happiness Score")
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
    logger.info("Saved GDP vs Happiness scatterplot")

    plt.clf()
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title('Correlation Matrix')
    plt.savefig('assignments_01/outputs/correlation_heatmap.png', bbox_inches='tight')
    logger.info('Saved correlation_heatmap.png')
    
    logger.info("All visual exploration completed!")

@task
def hypothesis_testing(df):
    logger = get_run_logger()
    logger.info('Starting hypothesis testing')
    
    h_2019 = df[df["Year"] == 2019]["happiness_score"].dropna()
    h_2020 = df[df["Year"] == 2020]["happiness_score"].dropna()

    t_stat, p_val = stats.ttest_ind(h_2019, h_2020)

    logger.info(f"Pandemic Test -> Mean 2019: {h_2019.mean():.3f}, Mean 2020: {h_2020.mean():.3f}")
    logger.info(f"Pandemic Test -> T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        logger.info("The pandemic caused a statistically significant change in global happiness.")
    else:
        logger.info("Surprisingly, the pandemic did NOT cause a statistically significant drop in global happiness scores.")
    
    russia_scores = df[df["Country"] == "Russia"]["happiness_score"].dropna()
    na_scores = df[df["Regional indicator"] == "North America and ANZ"]["happiness_score"].dropna()
    
    t_stat2, p_val2 = stats.ttest_ind(russia_scores, na_scores)
    
    logger.info(f"Custom Test -> Mean Russia: {russia_scores.mean():.3f}, Mean North America & ANZ: {na_scores.mean():.3f}")
    logger.info(f"Custom Test -> T-statistic: {t_stat2:.4f}, P-value: {p_val2:.4f}")
    
    if p_val2 < 0.05:
        logger.info("There is a statistically significant difference in happiness between Russia and North America/ANZ.")
    else:
        logger.info("There is no statistically significant difference in happiness between Russia and North America.")

@task
def check_correlations(df):
    logger = get_run_logger()
    logger.info("Starting Correlation and Multiple Comparisons...")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_remove = ["happiness_score", "Year", "Ranking"]
    explanatory_cols = [c for c in numeric_cols if c not in cols_to_remove]
    
    num_tests = len(explanatory_cols)
    adjusted_alpha = 0.05 / num_tests
    
    logger.info(f"Number of tests: {num_tests}")
    logger.info(f"Original alpha: 0.05. Bonferroni adjusted alpha: {adjusted_alpha:.4f}")
    
    df_clean = df.dropna(subset=["happiness_score"] + explanatory_cols)
    y = df_clean["happiness_score"]
    
    for col in explanatory_cols:
        x = df_clean[col]
        r, p_val = stats.pearsonr(x, y)
        
        sig_original = p_val < 0.05
        sig_adjusted = p_val < adjusted_alpha
        
        logger.info(f"--- Feature: {col} ---")
        logger.info(f"Pearson r: {r:.4f}, P-value: {p_val:.5f}")
        logger.info(f"Significant at 0.05? {sig_original}")
        logger.info(f"Significant at adjusted {adjusted_alpha:.4f}? {sig_adjusted}")

@task
def summary_report(df):
    logger = get_run_logger()
    logger.info("=== FINAL SUMMARY REPORT ===")
    
    total_countries = df["Country"].nunique()
    years = df["Year"].unique()
    logger.info(f"Dataset Scope: The dataset covers {len(years)} years ({years.min()}-{years.max()}) and includes {total_countries} unique countries.")
    
    region_means = df.groupby('Regional indicator')['happiness_score'].mean().sort_values(ascending=False)
    top_3 = region_means.head(3).index.tolist()
    bottom_3 = region_means.tail(3).index.tolist()
    
    logger.info(f"Top 3 Happiest Regions: {', '.join(top_3)}")
    logger.info(f"Bottom 3 Least Happy Regions: {', '.join(bottom_3)}")
    
    h_2019 = df[df["Year"] == 2019]["happiness_score"].dropna()
    h_2020 = df[df["Year"] == 2020]["happiness_score"].dropna()
    t_stat, p_val = stats.ttest_ind(h_2019, h_2020)
    
    if p_val < 0.05:
        logger.info("Pandemic Effect: The COVID-19 pandemic caused a statistically significant drop in global happiness.")
    else:
        logger.info("Pandemic Effect: Despite expectations, the onset of the COVID-19 pandemic did NOT cause a statistically significant drop in global happiness scores.")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    explanatory_cols = [c for c in numeric_cols if c not in ["happiness_score", "Year", "Ranking"]]
    
    df_clean = df.dropna(subset=["happiness_score"] + explanatory_cols)
    y = df_clean["happiness_score"]
    
    adjusted_alpha = 0.05 / len(explanatory_cols)
    best_feature, best_r = "", 0
    
    for col in explanatory_cols:
        r, p = stats.pearsonr(df_clean[col], y)
        if p < adjusted_alpha and r > best_r:
            best_r = r
            best_feature = col
            
    logger.info(f"Strongest Predictor: '{best_feature}' is the strongest driver of happiness (Pearson r = {best_r:.3f}), remaining highly significant even after strict Bonferroni correction.")
   
    
@flow(name="World Happiness Pipeline")
def happiness_pipeline():
    merged_df = load_merge_data()
    descriptive_statistics(merged_df)
    visualize_data(merged_df)
    hypothesis_testing(merged_df)
    check_correlations(merged_df)
    summary_report(merged_df) 

if __name__ == "__main__":
    happiness_pipeline()