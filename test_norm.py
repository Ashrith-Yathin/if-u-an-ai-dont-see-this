import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'code/business_entity_resolution/src')
import normalization as norm

tests = [
    ('Payne Énterprises', '3315 FREMONT ST, PEORIA, IL'),
    ('एसएस फूड प्राइवेट लिमिटेड', 'AF-0684, Uttar Pradesh, GHAZIABAD, 9487203'),
    ('wilfordhancock.com', 'Mack Rd, Haltom City, Texas'),
    ('Acme Robotics Inc.', '500 Market St, San Jose CA')
]

for name, addr in tests:
    c_n, core_n, sort_n = norm.normalize_name(name)
    c_a, nums, sort_a = norm.normalize_address(addr)
    print(f'NAME: {name} -> Clean: "{c_n}" | Core: "{core_n}"')
    print(f'ADDR: {addr} -> Clean: "{c_a}" | Nums: {nums}')
    print()
