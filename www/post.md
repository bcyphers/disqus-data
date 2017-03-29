<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css"
integrity="sha384-rwoIResjU2yc3z8GV/NPeZWAv56rSmLldC3R/AZzGRnGxQQKnKkoFVhFQhNUwEyJ"
crossorigin="anonymous">
<link rel="stylesheet" href="styles.css">

<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://code.jquery.com/jquery-3.1.1.min.js"></script>
<script
src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js"
integrity="sha384-DztdAPBWPRXSA/3eYEEUWrWCy7G5KFbe8fFjk5JAIxUYHKkDx6Qin1DkWx51bBrb"
crossorigin="anonymous"></script>
<script
src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js"
integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn"
crossorigin="anonymous"></script>

# Mapping the Disqus Universe, Part 1

Disqus is a comment service for websites. You've probably seen their ubiquitous little speech bubble around the web; maybe you even have an account. The company powers sites like The Atlantic, Rolling Stone, People, and Breitbart (although only three of those are advertised on their website). According to their literature, they serve 2 billion monthly unique users and 50 million comments per month from 196 countries. Disqus calls itself "the world's largest first-party data set."

Recently, I got curious about how users from various parts of their network interact with each other. Luckily, they have a fantastic, free public API; it took me about 5 minutes to get a private key and start pulling down data. Here's what I did and what I found.

### Getting Started

Most of my interactions with Disqus are via the Atlantic. I noticed that comments there represent a more diverse set of political views than comments on other mainstream, left-leaning sites, like the New York Times and the Washington Post. One difference is that the Times and the Post developed and deploy their own comment systems, while the Atlantic uses Disqus. One day, scanning a thread of fervently pro-Trump comments on an obviously Trump-critical piece, I wondered, where are these people coming from? Thanks to Disqus, I could find out. For example, a user who commented "White privilege is a racist canard" turned out to be quite active on The Federalist, a reliably conservative blog. And if I could check for one user, why not look at hundreds of them? If I could look at one site, why not look at them all?

*On privacy*: I was concerned with trends across sites, not with the habits of any individual user. I won't call out anyone by name (real or digital), and I'll keep direct quotes to a minimum. All the information I've pulled is publically available, both via the Disqus API and in comments sections around the web.  Nevertheless, I have downloaded several hundred megabytes of user data, which makes me kinda uneasy. Time and again, [research](https://arxiv.org/pdf/0903.3276.pdf) [has](http://hgi.ruhr-uni-bochum.de/media/emma/veroeffentlichungen/2011/06/07/deanonymizeSN-Oakland10.pdf) [shown](https://www.cs.cornell.edu/~shmat/shmat_oak08netflix.pdf) that an aggregate of seemingly innocuous data points about a person can be used to reveal tremendous amounts of information, and I don't want to dox anyone. I encourage any concerned Disqus users to [make their profiles private](https://help.disqus.com/customer/portal/articles/1197204-making-your-activity-private) in order to make it harder (though [not impossible](https://pdfs.semanticscholar.org/5697/02d3f854ecd55d3877d2b6cb45292aa7ae29.pdf)) for people like me to snoop.  

So, into the data. Most of the data gathered for this post are from the
[`forum.listMostActiveUsers`](https://disqus.com/api/docs/forums/listMostActiveUsers/)
and
[`user.listMostActiveForums`](https://disqus.com/api/docs/users/listMostActiveForums/)
endpoints. In Disqus jargon, a "forum" is more or less equivalent to a site,
comments on one article are grouped in a "thread," and each comment is a "post."
`forum.listMostActiveUsers` gives a list of a few hundred top users for a
particular forum. For each of those, I called `user.listMostActiveForums`, which
returns a list of the forums that a particular user frequents most. By doing so,
I could generate a vector for each site showing which other sites its top users visited most. For example, here's the number of the Atlantic's top users (349) for whom each site is a "top forum":

```python
outgoing_links["The Atlantic"] = {
"The Hill": 149, 
"Bloomberg View": 140, 
"Mother Jones": 122, 
"Breitbart News Network": 104, 
...
}
```

Several thousand forums are referenced in the full vector, most of them by just one user. 

Now, let's build and normalize this vector for each forum, and compare how their users interact with other popular forums:

|              | The Hill | Bloomberg View | InfoWars | National Review | Mother Jones |
|--------------|----------|----------------|----------|-----------------|--------------|
| The Atlantic | 0.30     | 0.28           | 0.06     | 0.20            | 0.25         |
| Breitbart    | 0.64     | 0.15           | 0.62     | 0.12            | 0.06         |

<p class="caption">What fraction of top users from forum A (vertical axis) frequent forum B (horizontal)?</p>

Right off the bat, we have some interesting data. A bigger portion of The Atlantic's top users frequent the conservative stalwart National Review than do Breitbart's. Almost two-thirds of Breitbart's top users frequent Alex Jones' Infowars. And The Hill seems to be popular with everyone.

That's cool, but it's tough to interpret. I found some forums' top users were very promiscuous: they would all frequent dozens of other sites. Other forums attracted an audience with more exclusive taste. And just because the top users of The Atlantic also visit Breitbart doesn't mean the inverse is true: Brietbart's top users might not touch that liberal rag. 

### Correlations

It would be nice to get a straightforward measure of how similar two sites are. Let's try the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between each pair of forums. The cross-pollination vectors are very long, so correlation values should reflect, roughly, the similarity between the habits of each forum's top users. This makes another fun matrix:

|              | The Atlantic | Breitbart | The Hill | The AV Club |
|--------------|--------------|-----------|----------|-------------|
| The Atlantic | 1            | 0.57      | 0.72     | 0.25        |
| Breitbart    | 0.57         | 1         | 0.86     | 0.07        |
| The Hill     | 0.72         | 0.86      | 1        | 0.12        |
| The AV Club  | 0.25         | 0.07      | 0.12     | 1           |

<p class="caption">Example correlations between user cross-pollination vectors
on the top 500 most-referenced forums</p>

Disqus forums are labeled with one of a few categories, like "News," "Sports,"
and "Business." Most of the sites I pulled data for are news- or
politics-related, and those tend to correlate highly with each other in spite of
ideological differences. Forums from under-represented categories, like
Entertainment and Technology, correlate less strongly with anything (but more
strongly with forums in their own category). It seems like the numbers are more
meaningful when compared against a specific forum than they are globally. *The
Atlantic* correlates with *Breitbart* (0.57), but 154 other forums correlate with
*Breitbart* more strongly; likewise, 167 forums are more correlated with *The
Atlantic*. Meanwhile, the top correlation with *The AV Club* is *Tom and Lorenzo* at 0.57.


Here's a broader view of the correlations between the top 100 most referenced sites:

<div class="matrix"></div>
<script type="text/javascript" src="corr_matrix.js"></script>

<p class="caption">Correlation matrix between top 100 referenced sites</p>

### Clustering

Let's see if we can make more sense of the network. 

A correlation matrix looks an awful lot like a fully-connected graph, and graphs lend themselves to all kinds of cool analysis. Let's say each forum is a vertex, and each positive correlation value is an edge. We can use the correlation values to power a force-directed graph, where a high correlation pulls two forums together and a weak or negative one pushes them apart. With a few hundred forums, it looks like this:

<div class="container">
  <div class="row">
    <svg id="force-directed" width="1280" height="720" class="img-fluid" alt="Responsive image"></svg>
  </div>
  <div class="row info-header">
    <div class="col">
      <h4 id="forum-name">Forum</h4>
    </div>
    <div class="col-3">
      <div class="row">
        <div class="dropdown">
          <button class="btn btn-primary dropdown-toggle" type="button"
  data-toggle="dropdown">Color by...<span class="caret"></span></button>
          <ul class="dropdown-menu" id='coloring-select'>
            <li><a href="#" value='category'>Categories</a></li>
            <li><a href="#" value='group'>Markov clusters</a></li>
            <li><a href="#" value='activity'>Activity</a></li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  <div class="row info-container">
    <div class="col-5">
      <ul class="list-group" id="details-ul">
        <li id="detail-url">Details</li>
        <li id="detail-category"></li>
        <li id="detail-cluster"></li>
        <li id="detail-activity"></li>
        <li id="detail-connectivity"></li>
        <li id="detail-description"></li>
      </ul>
    </div>
    <div class="col-3">
      <b id="correlations-title">Correlates with</b>
      <ol class="list-group" id="correlate-ol">
        <li id="correlation-0"></li>
        <li id="correlation-1"></li>
        <li id="correlation-2"></li>
        <li id="correlation-3"></li>
        <li id="correlation-4"></li>
      </ol>
    </div>
    <div class="col">
      <b id="topics-title">Top topics</b>
      <ol class="list-group" id="topics-ol">
        <li id="topic-0" class="row">
          <div class="col-2 topic-score"></div>
          <div class="col topic-name"></div>
        </li>
        <li id="topic-1" class="row">
          <div class="col-2 topic-score"></div>
          <div class="col topic-name"></div>
        </li>
        <li id="topic-2" class="row">
          <div class="col-2 topic-score"></div>
          <div class="col topic-name"></div>
        </li>
      </ol>
    </div>
  </div>
</div>
<script src="d3-vis.js"></script>
<script src="interactive-interface.js"></script>

Each circle is a forum, and links are correlations. In this graph, I only included correlations greater than 0.2, and only the top five correlations for each forum. Link strength is based on correlation strength. By default, the color of each circle corresponds to its Disqus category.  

Markov Clustering (MCL) is one way to group the forums algorithmically. You can read about it [here](http://micans.org/mcl/), it's fascinating. Basically, you provide a graph with edge weights, and the algorithm manipulates the graph so that each vertex "clusters" around a single other vertex -- possibly itself. There are two parameters, e and r, which control how large the clusters are. You can see the results of the clustering by selecting "Markov clusters" under "Color by..."

What's striking about the graph is how tight some of the regions are. Again, the
"News" sites tend to cluster together in the middle, but that's not the whole
story. If you scan that big region in the middle, you'll notice a lot of
"Culture," "Business," and some "Entertainment" as well. And check out the
region around Breitbart (one of the biggest circles, to the left of the graph).
There are dozens of sites that correlate quite strongly with each other --
the conservative blogosphere. 

Turn on Markov coloring. You can click on the "Clusters with" link under forum
details to hilight a cluster in the graph. Breitbart & co. are mostly in blue,
though the green group is closely related. Liberal blogs have their own clusters
-- Media Matters, Wonkette, and Raw Story form the core of a tan group towards
the bottom. The orange group includes a lot of the "old guard:" The Atlantic,
Rolling Stone, and CBS local affiliates, plus, for some reason, The AV Club and
a whole bunch of entertainment sites. 

Look at the top correlations for blue sites. Most of them have more than one
correlation above .9; Breitbart has five.  The green group is even more
tight-knit. Due to the way I added links (a maximum of five per forum), a
portion of the green group cut itself off from the rest of the network by being
*too* connected. You should see it floating off to the right. TotalConservative,
UnfilteredPatriot, PatriotNewsDaily: all these forums have r-values of .99
with each other. They also correlate with blue-group forums quite well, but
don't get a chance to link to them because of link limits. The only other
cliques in the graph this tight involve sub-forums of the same site, e.g. CBS
local stations or Disqus channels.

<p class="caption">[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) plot of correlation vectors</p>

#### Fifty thousand people used to post here. Now it's a ghost town.

Let me direct your attention to another trend. Color the graph by "activity," and check out the forums that don't have any recent posts. Any of them sound familiar? CNN, NBC, Bloomberg, Politico... that's not a glitch. Dozens of sites have shut down their Disqus forums in the past couple of years, and a good chunk of them are "mainstream." 

Some of them, like Politico and The National Review, have switched over to Facebook-powered comment machinery. Others, like CNN and Bloomberg, appear to have ditched comments altogether. It's possible this is related to hate speech issues Disqus has had in the past (and have recently [tried to address](https://blog.disqus.com/our-commitment-to-fighting-hate-speech)), or maybe it's part of [the Left's war on comments](http://www.breitbart.com/tech/2015/10/27/the-lefts-war-on-comment-sections/)... I haven't delved deeper, but it's interesting either way.


### d
