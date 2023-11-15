
def extract_same_range_from_climate_data(health_data, climate_dataset):
    first, last = min(health_data['periodname']), max(health_data['periodname'])
    # add month to datetime
    last = last.replace(month=last.month+1)
    return climate_dataset[str(first):str(last)]