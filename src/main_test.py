import numpy



'''
n_pool = 60
idxs_lb = numpy.ones(n_pool, dtype=bool)
idxs_lb[0] = False
idxs_lb[1] = False
idxs_lb[2] = False

idxs_unlabeled = numpy.arange(n_pool)[~idxs_lb]
print(idxs_unlabeled)
xt = numpy.ones((60, 10, 100))
print(xt[idxs_unlabeled])


if __name__ == "__main__":
    print("OOOOOOOPXFUNGE")

    emb = numpy.zeros([10, 100, 22])

    print(emb.shape)
    print(emb.shape)


import numpy as np

def generate_random_list(n):
    """Generates a list of n random values between 0 and 1 using NumPy."""
    random_list = np.random.rand(n)
    print(type(random_list))
    return random_list

# Example usage
n = 5
random_list = generate_random_list(n)
print(type(random_list))
print(random_list)
print(type(random_list))
'''
import torch

list1 = [5, 2, 8, 1]  # List used for sorting
list2 = [torch.tensor([1.5, 2.3]), torch.tensor([5.1]), torch.tensor([0.8, 3.2, 1.1]), torch.tensor([9.3])]
list3 = [3.14, 7.2, 0.55, 9.8]

combined_data = list(zip(list1, list2, list3))
sorted_data = sorted(combined_data)

ordered_list1, ordered_list2, ordered_list3 = zip(*sorted_data)

print(ordered_list1)
print(ordered_list2)
print(ordered_list3)
