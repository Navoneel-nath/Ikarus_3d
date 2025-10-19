import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot styles for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
try:
    df = pd.read_csv('intern_data_ikarus.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'intern_data_ikarus.csv' not found.")
    print("Please make sure you have downloaded the dataset and placed it in the correct directory.")
    df = pd.DataFrame() # Create an empty dataframe to avoid further errors

if not df.empty:
    # Display the first few rows and basic info
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()

# A clean dataset is essential for building a reliable model. We will handle missing values according to the project requirements.

if not df.empty:
    # The 'images' column is critical for our multi-modal model.
    # We will drop any rows where the image URL is missing.
    
    print(f"Number of rows before dropping missing images: {len(df)}")
    df.dropna(subset=['images'], inplace=True)
    print(f"Number of rows after dropping missing images: {len(df)}")
    # For products without a description, we'll create a basic one 
    # by combining the title and brand. This ensures the text embedding
    # model has some content to work with.
    
    missing_desc_count = df['description'].isnull().sum()
    print(f"\nNumber of rows with missing description: {missing_desc_count}")
    
    # Fill missing descriptions
    df['description'] = df.apply(
        lambda row: f"{row['title']} by {row['brand']}" if pd.isnull(row['description']) else row['description'],
        axis=1
    )
    
    print("Missing descriptions have been filled.")
    
    # The requirement is to keep rows with missing prices. We'll just check how many there are.
    # For the purpose of analysis, we can fill them with 0 or the median, but for now, we just acknowledge them.
    missing_price_count = df['price'].isnull().sum()
    print(f"\nNumber of rows with missing price: {missing_price_count}")

    # For visualization purposes, let's convert the price to a numeric type, coercing errors.
    # The price column seems to have '$' and commas, so we need to clean it first.
    
    def clean_price(price):
        if isinstance(price, str):
            try:
                # Remove '$', ',', and other non-numeric characters, then convert to float
                return float(price.replace('$', '').replace(',', '').strip())
            except ValueError:
                return np.nan # Return NaN if conversion fails
        return price

    df['price_cleaned'] = df['price'].apply(clean_price)
    
    print("\nCleaned 'price' column and created 'price_cleaned'.")

# Now that the data is clean, let's visualize it to understand the distributions.

if not df.empty:
    # Let's see which categories are the most common in our dataset.
    
    plt.figure(figsize=(15, 8))
    # Taking the top 20 categories for better readability
    top_categories = df['categories'].value_counts().nlargest(20)
    sns.barplot(y=top_categories.index, x=top_categories.values, palette='viridis')
    plt.title('Top 20 Product Categories Distribution', fontsize=16)
    plt.xlabel('Number of Products', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Now, let's analyze the distribution of brands.
    
    plt.figure(figsize=(15, 8))
    # Taking the top 20 brands
    top_brands = df['brand'].value_counts().nlargest(20)
    sns.barplot(y=top_brands.index, x=top_brands.values, palette='plasma')
    plt.title('Top 20 Product Brands Distribution', fontsize=16)
    plt.xlabel('Number of Products', fontsize=12)
    plt.ylabel('Brand', fontsize=12)
    plt.tight_layout()
    plt.show()

    # We will use the 'price_cleaned' column for this. A histogram is a good way to see the price distribution.
    # We will filter out extreme outliers for a more meaningful plot.
    
    plt.figure(figsize=(12, 6))
    # Let's cap the price at a reasonable amount (e.g., 99th percentile) to avoid skewed plots
    price_cap = df['price_cleaned'].quantile(0.99)
    filtered_prices = df['price_cleaned'][df['price_cleaned'] <= price_cap]
    
    sns.histplot(filtered_prices, bins=50, kde=True, color='blue')
    plt.title(f'Product Price Distribution (up to ${price_cap:.2f})', fontsize=16)
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

if not df.empty:
    df.to_csv('cleaned_intern_data.csv', index=False)
    print("\nCleaned data saved to 'cleaned_intern_data.csv'.")
    
print("\n--- Data Analytics & Preparation Complete ---")