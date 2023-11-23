import hashlib

def flajolet_martin(data):
    max_rightmost_zero_bit_position = -1
    
    for element in data:
        hash_value = hashlib.md5(element.encode()).hexdigest()
        rightmost_zero_bit_position = len(hash_value) - len(hash_value.rstrip('0'))
        
        if rightmost_zero_bit_position > max_rightmost_zero_bit_position:
            max_rightmost_zero_bit_position = rightmost_zero_bit_position
    
    estimated_number_of_distinct_elements = 2 ** max_rightmost_zero_bit_position
    
    return estimated_number_of_distinct_elements

data = "ab23 Big data is a field that treats ways to analyze from or otherwise deal with data Current usage of the term big data tends to predictive analytics user behavior, analytics or advanced data analytics Data sets grow rapidly in part because they are increasingly Internet of things devices such as mobile devices. Xz33. Small Small Small Small Small Small Small Small Small Small Small Small Small Small Small Small"

elements = data.split()

estimated_count = flajolet_martin(elements)
small_count = elements.count("Small")

print("Số tên của các phần tử riêng biệt là:", estimated_count)
