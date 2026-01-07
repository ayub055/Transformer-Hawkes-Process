import sys
import os
sys.path.append('.')

from collections import defaultdict
from preprocess.Dataset import convert_to_hawkes_format

if __name__ == "__main__":
    data = [
        {
            'customer': 'C1',
            'loan_type_seq': 'HL|TWD|PL',
            'loan_amt_seq': '2000000|200000|500000',
            'age_seq': '42|45|47'
        }
    ]

    event_streams, num_types = convert_to_hawkes_format(data)

    print(f"Number of event types: {num_types}")
    print(f"Number of customers: {len(event_streams)}")

    for i, events in enumerate(event_streams):
        print(f"\nCustomer {data[i]['customer']}:")
        for j, event in enumerate(events):
            print(f"  Event {j+1}: type={event['type_event']}, amount={event['loan_amount']:.0f}, time_since_start={event['time_since_start']:.1f}, time_gap={event['time_since_last_event']:.1f}")
