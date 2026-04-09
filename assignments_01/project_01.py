import pandas as pd
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns

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

@flow(name="World Happiness Pipeline")
def happiness_pipeline():
    merged_df = load_merge_data()
    descriptive_statistics(merged_df)
    visualize_data(merged_df)

if __name__ == "__main__":
    happiness_pipeline()