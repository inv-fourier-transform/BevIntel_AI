import pandas as pd

def prepare_features(raw_input: dict, model_data: dict):
    """
    raw_input: dict with keys matching Streamlit input names, e.g.
      {
        'age': 30,
        'gender': 'M',
        'zone': 'Urban',
        'occupation': 'Working Professional',
        'income_level': '16L - 25L',
        'consume_frequency': '3-4 times',
        'current_brand': 'Newcomer',
        'preferable_consumption_size': 'Medium (500 ml)',
        'awareness_of_other_brands': '2 to 4',
        'reasons_for_choosing_brands': 'Quality',
        'flavor_preference': 'Traditional',
        'purchase_channel': 'Retail Store',
        'packaging_preference': 'Premium',
        'health_concerns': 'Medium (Moderately health-conscious)',
        'typical_consumption_situations': 'Casual (eg. At home)'
      }
    model_data: the dict saved with joblib, containing mappings, feature_names, etc.
    """

    mappings = model_data['mappings']
    feat_names = model_data['feature_names']

    # Step 1: start building a dict for final features, initialize with zeros
    data = {c: 0 for c in feat_names}

    # Step 2: numeric score features from mappings
    cf = mappings['cf_mapping'][ raw_input['consume_frequency'] ]
    ab = mappings['ab_mapping'][ raw_input['awareness_of_other_brands'] ]
    zone_score = mappings['zone_mapping'][ raw_input['zone'] ]
    income_score = mappings['income_mapping'][ raw_input['income_level'] ]

    data['cf_score'] = cf
    data['ab_score'] = ab
    data['cf_ab_score'] = cf / (cf + ab) if (cf + ab) > 0 else 0
    data['zone_score'] = zone_score
    data['income_score'] = income_score
    data['zas_score'] = zone_score * income_score

    # Step 3: age → age_group (binning) → label-encode
    age = raw_input['age']
    bins = mappings['age_bins']['bins']
    labels = mappings['age_bins']['labels']
    age_group = pd.cut([age], bins=bins, labels=labels, right=True)[0]
    # convert to label-encoded integer
    data['age_group'] = mappings['label_encoding']['age_group'][age_group]

    # Step 4: label-encode size and health concerns
    data['preferable_consumption_size'] = mappings['label_encoding']['preferable_consumption_size'][ raw_input['preferable_consumption_size'] ]
    data['health_concerns'] = mappings['label_encoding']['health_concerns'][ raw_input['health_concerns'] ]

    # Step 5: derived boolean / binary feature 'bsi'
    # bsi logic: 1 if (current_brand ≠ "Established") AND (reason in ['Price','Quality']), else 0
    brand = raw_input['current_brand']
    reason = raw_input['reasons_for_choosing_brands']
    if (brand != "Established") and (reason in ['Price', 'Quality']):
        data['bsi'] = 1
    else:
        data['bsi'] = 0

    # Step 6: one-hot / dummy encoding for categorical vars - set the appropriate dummy to 1
    # List of one-hot columns as saved
    one_hot_cols = mappings['one_hot_columns']

    # For each possible dummy column, check if it corresponds to raw_input, set to 1 if yes
    # gender
    if raw_input['gender'] == 'M':
        data['gender_M'] = 1
    # occupation
    occ = raw_input['occupation']
    if occ == 'Retired':
        data['occupation_Retired'] = 1
    elif occ == 'Student':
        data['occupation_Student'] = 1
    elif occ == 'Working Professional':
        data['occupation_Working Professional'] = 1
    # current_brand
    if raw_input['current_brand'] == 'Newcomer':
        data['current_brand_Newcomer'] = 1
    # reasons_for_choosing_brands
    reason = raw_input['reasons_for_choosing_brands']
    if reason == 'Brand Reputation':
        data['reasons_for_choosing_brands_Brand Reputation'] = 1
    elif reason == 'Price':
        data['reasons_for_choosing_brands_Price'] = 1
    elif reason == 'Quality':
        data['reasons_for_choosing_brands_Quality'] = 1
    # flavor_preference
    if raw_input['flavor_preference'] == 'Traditional':
        data['flavor_preference_Traditional'] = 1
    # purchase_channel
    if raw_input['purchase_channel'] == 'Retail Store':
        data['purchase_channel_Retail Store'] = 1
    # packaging_preference
    pkg = raw_input['packaging_preference']
    if pkg == 'Premium':
        data['packaging_preference_Premium'] = 1
    elif pkg == 'Simple':
        data['packaging_preference_Simple'] = 1
    # typical_consumption_situations
    sit = raw_input['typical_consumption_situations']
    if sit == 'Casual (eg. At home)':
        data['typical_consumption_situations_Casual (eg. At home)'] = 1
    elif sit == 'Social (eg. Parties)':
        data['typical_consumption_situations_Social (eg. Parties)'] = 1

    # Build a 1-row df
    df = pd.DataFrame([data], columns=feat_names)
    return df

# Reverse mapping for the target price
rev_mappings_target = {
     0 : '₹50-₹100',
     1 : '₹100-₹150',
     2 : '₹150-₹200',
     3 : '₹200-₹250'
}

def reverse_mapping(pred):
    return rev_mappings_target.get(pred[0])