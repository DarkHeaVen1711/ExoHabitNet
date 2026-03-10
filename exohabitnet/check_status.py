import pandas as pd

df = pd.read_csv('data/data_collection_log.csv')
success_df = df[df['collection_status'] == 'SUCCESS']

print('\n' + '='*50)
print('DATA COLLECTION STATUS')
print('='*50)
print(f'\nTotal records: {len(df)}')
print(f'Successful collections: {len(success_df)}')

print('\nClass Distribution:')
print(success_df['label'].value_counts())

print('\nClass Percentages:')
for label, count in success_df['label'].value_counts().items():
    pct = (count / len(success_df)) * 100
    print(f'  {label:20s}: {count:4d} samples ({pct:5.1f}%)')

print('\n' + '='*50)
