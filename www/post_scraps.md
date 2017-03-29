
how I did it:
I then sorted the linked sites by number of links and pulled data from the top one. Re-sort by total user links, rinse, repeat. After I'd gathered references to a few hundred forums like this, I started calling  thread.listPopular for each forum and pulling data for those with the most activity in their 100 most popular threads.

After running my script for a while, I had data on the top users from a few hundred forums. For each one of those, I had a vector representing the number of "connections" from that forum to each of a few thousand other forums. If we divide each forum's vector by its total number of users, we can compare the vectors for different sites to see how similar they are:


