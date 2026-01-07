import csv
import random
import pickle
from collections import defaultdict
from preprocess.Dataset import convert_to_hawkes_format, save_data

def generate_customer_data(num_customers=1000):
    """Generate synthetic customer loan data in the required format."""

    loan_types = ['HL', 'TWD', 'PL', 'AL', 'BL', 'CL', 'DL']

    customers_data = []

    for i in range(num_customers):
        num_loans = random.randint(1, 15)
        ages = []
        current_age = random.uniform(25, 52)
        ages.append(current_age)

        for _ in range(num_loans - 1):
            time_gap = random.uniform(1, 5)  # 1-5 years between loans (guarantees non-decreasing ages)
            current_age += time_gap
            ages.append(current_age)

        # Verify ages are non-decreasing (constraint for Hawkes process)
        assert all(ages[i] <= ages[i+1] for i in range(len(ages)-1)), f"Ages not non-decreasing: {ages}"

        # Generate loan types and amounts
        loan_type_list = []
        loan_amt_list = []

        for j in range(num_loans):
            # Slightly favor certain loan types for realism
            if j == 0:
                # First loan often HL or PL
                loan_type = random.choice(['HL', 'PL', 'AL'])
            else:
                loan_type = random.choice(loan_types)

            loan_type_list.append(loan_type)

            # Generate realistic loan amounts based on loan type
            if loan_type == 'HL':
                amount = random.randint(500000, 5000000)  # Home loans: larger amounts
            elif loan_type == 'PL':
                amount = random.randint(100000, 1000000)  # Personal loans: medium amounts
            elif loan_type == 'AL':
                amount = random.randint(200000, 2000000)  # Auto loans: medium-large
            else:
                amount = random.randint(50000, 500000)   # Other loans: smaller amounts

            loan_amt_list.append(amount)

        # Format the data
        customer_data = {
            'customer': f'C{i+1:04d}',  # C0001, C0002, etc.
            'loan_type_seq': '|'.join(loan_type_list),
            'loan_amt_seq': '|'.join(map(str, loan_amt_list)),
            'age_seq': '|'.join(f'{age:.1f}' for age in ages)
        }

        customers_data.append(customer_data)

    return customers_data



def save_train_dev_test_data(train_data, dev_data, test_data, num_types, output_dir='./'):
    """Save train, dev, and test data to separate pickle files."""
    import os
    save_data({'train': train_data, 'dev': [], 'test': []}, num_types,
              os.path.join(output_dir, 'train.pkl'))

    save_data({'train': [], 'dev': dev_data, 'test': []}, num_types,
              os.path.join(output_dir, 'dev.pkl'))

    save_data({'train': [], 'dev': [], 'test': test_data}, num_types,
              os.path.join(output_dir, 'test.pkl'))

    print(f"Saved train.pkl ({len(train_data)} samples), dev.pkl ({len(dev_data)} samples), test.pkl ({len(test_data)} samples)")

def generate_and_save_hawkes_data(num_samples=2000, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    """Generate customer data, convert to Hawkes format, and save to separate train/dev/test pickle files."""
    print(f"Generating {num_samples} customer loan records...")
    data = generate_customer_data(num_samples)

    print("Converting to Hawkes format...")
    event_streams, num_types = convert_to_hawkes_format(data)

    # Split the data
    n_total = len(event_streams)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    n_test = n_total - n_train - n_dev  # Ensure all samples are used

    train_data = event_streams[:n_train]
    dev_data = event_streams[n_train:n_train + n_dev]
    test_data = event_streams[n_train + n_dev:]

    print("Saving to separate pickle files...")
    save_train_dev_test_data(train_data, dev_data, test_data, num_types)

    print(f"\n✓ Total: {n_total} samples")
    print(f"✓ Number of event types: {num_types}")
    print(f"✓ Train/Dev/Test split: {train_ratio:.1%}/{dev_ratio:.1%}/{test_ratio:.1%}")

if __name__ == "__main__":
    generate_and_save_hawkes_data()