# Profiling

This module is for figuring out how efficient my methods are. Two functions are analysed here: the one computing closest pairs, and the r-swoosh algorithm.

## Closest Pairs

Finding the closest pair between two sets of points can be subquadratically using a 2D-tree. However, the first question may arise, what's better, indexing the biggest set or the smallest? Profiling showed that it is indeed better to use the biggest one by orders of magnitude when the difference in size is big enough. The reason for that is the overhead introduced in calling the `query()` function. Indexing the bigger set makes each query more costly, but it reduces the number of queries made. Many calls to a function that is cheap is more expensive than few calls to a function that is expensive.

Finding the N closest pairs involves yet another problem, how to store the N pairs during the execution of the algorithm. Asymptotically, the both approaches I will mention are equal. However, one of them was orders of magnitude better than the other. The main operation is to update a given data structure of N values with N new values so that we maintain the N lowest ones. One way is to use a priority queue to store the values. The solution is then to remove the biggest element of the queue, compare it with one of the elements in the new set and add the smallest. Cost: O(Nlog(N)). A second way is to merge both sets in one list, sort that list and retrieve the first N elements. Cost: O(Nlog(N)). Surprisingly, the second approach is way better. There is yet another solution that is O(N) but it requires the new set to be sorted and it is not so simple so it was ignored.

## R-Swoosh

Profiling this function was useful to know the expected execution of this algorithm. It is one or two seconds for reasonable amounts of cells. No important discovery was made here about any improvement to make.
