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
<script type="text/javascript" src="topic-vis.js"></script>

# Mapping the Disqus Universe, Part 1

Disqus is a comment service for websites. You've probably seen their ubiquitous little speech bubble around the web; maybe you even have an account. The company powers sites like The Atlantic, Rolling Stone, People, and Breitbart (although only three of those are advertised on their website). According to their literature, they serve 2 billion monthly unique users and 50 million comments per month from 196 countries. Disqus calls itself "the world's largest first-party data set."

Recently, I got curious about how users from various parts of their network interact with each other. Luckily, they have a fantastic, free public API; it took me about 5 minutes to get a private key and start pulling down data. Here's what I did and what I found.

### Getting started

Most of my interactions with Disqus are via [The Atlantic](https://www.theatlantic.com/). At some point, I noticed that comments there represent a more diverse set of political views than comments on other mainstream, left-leaning sites, like the New York Times and the Washington Post. One day, scanning a thread of fervently pro-Trump comments on a piece obviously critical of 45, I wondered, where are these people coming from? Thanks to Disqus, I could find out. For example, a user who commented "White privilege is a racist canard" turned out to be quite active on The Federalist, a reliably conservative blog. And if I could check for one user, why not check hundreds of them? If I could look at one site, why not look at them all?

<img src="./media/user-profile.png" alt="Disqus profile page" width="50%" class="centered">
<p class="caption">A sample Disqus profile page</p>


*A note on privacy*: I was interested in trends across Disqus forums, not with the habits of any individual user. In this post I won't call out anyone by name, real or digital, and I'll keep direct quotes to a minimum. All the information I've pulled is publically available, both via the Disqus API and in comments sections around the web. Nevertheless, I've downloaded several hundred megabytes of user data. Time and again, [research](https://arxiv.org/pdf/0903.3276.pdf) [has](https://www.media.mit.edu/research/highlights/unique-shopping-mall-reidentifiability-credit-card-metadata) [shown](https://www.cs.cornell.edu/~shmat/shmat_oak08netflix.pdf) that a collection of seemingly innocuous data points about a person can be used to reveal tremendous amounts of information, and I don't want to dox anyone. I encourage any concerned Disqus users to [make their profiles private](https://help.disqus.com/customer/portal/articles/1197204-making-your-activity-private) in order to make it harder (though [not impossible](https://pdfs.semanticscholar.org/5697/02d3f854ecd55d3877d2b6cb45292aa7ae29.pdf)) for people like me to snoop.  

So, into the data. Most of the data gathered for this post are from the [`forum.listMostActiveUsers`](https://disqus.com/api/docs/forums/listMostActiveUsers/) and [`user.listMostActiveForums`](https://disqus.com/api/docs/users/listMostActiveForums/) endpoints. In Disqus jargon, a "forum" is more or less equivalent to a site, comments on one article are grouped in a "thread," and each comment is a "post." `listMostActiveUsers` gives a list of a few hundred top users for a particular forum. For each of those, I called `listMostActiveForums`, which returns a list of the forums that a particular user frequents most. By doing so, I could generate a vector for each site showing which other sites its top users visited most. For example, here's the number of the Atlantic's top users (349) for whom each site is a "top forum":

```python
outgoing_links["The Atlantic"] = {
"The Hill": 149, 
"Bloomberg View": 140, 
"Mother Jones": 122, 
"Breitbart News Network": 104, 
...
}
```

Several thousand forums are referenced in the full vector, most of them by just one user. The full dataset I've collected has vectors like this for 444 different forums.

If we normalize these vectors by the number of top users for each forum, we can compare how their users interact with other popular forums:

|              | The Hill | Bloomberg View | InfoWars | National Review | Mother Jones |
|--------------|----------|----------------|----------|-----------------|--------------|
| The Atlantic | 0.30     | 0.28           | 0.06     | 0.20            | 0.25         |
| Breitbart    | 0.64     | 0.15           | 0.62     | 0.12            | 0.06         |

<p class="caption">What fraction of top users from forum A (vertical axis) frequent forum B (horizontal)?</p>

Right off the bat, there's some interesting data. A bigger portion of The Atlantic's top users frequent the conservative stalwart [National Review](http://www.nationalreview.com/) than do [Breitbart](http://www.breitbart.com/)'s. Almost two-thirds of Breitbart's top users frequent Alex Jones' [Infowars](https://www.infowars.com/). And [The Hill](http://thehill.com/) seems to be popular with everyone.

That's neat, but it's tough to interpret. I found some forums' top users were rather promiscuous, and would all frequent dozens of other sites. Other forums attracted an audience with more exclusive taste. And just because the top users of The Atlantic also visit Breitbart doesn't mean the inverse is true; all of Brietbart's top users might stay away from that liberal rag. 

### Correlations

It would be nice to get a straightforward measure of how similar two sites are.  Let's try the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between each pair of forums. The cross-pollination vectors are very long, so correlation values should reflect, roughly, the similarity between the habits of each forum's top users. This makes another fun matrix:

|              | The Atlantic | Breitbart | The Hill | The AV Club |
|--------------|--------------|-----------|----------|-------------|
| The Atlantic | 1            | 0.57      | 0.72     | 0.25        |
| Breitbart    | 0.57         | 1         | 0.86     | 0.07        |
| The Hill     | 0.72         | 0.86      | 1        | 0.12        |
| The AV Club  | 0.25         | 0.07      | 0.12     | 1           |

<p class="caption">Example correlations between user cross-pollination vectors
on the top 500 most-referenced forums</p>

Cool! That... kind of makes sense. As a news site, The Atlantic is similar to both Breitbart and The Hill, but probably more similar to The Hill. And [The AV Club](http://avclub.com/) is not that similar to any of them, but its left-leaning audience is probably most similar to The Atlantic's.

Disqus forums are labeled with one of a few categories, like "News," "Sports," and "Business." Most of the sites I pulled data for are news- or politics-related, and those tend to correlate highly with each other in spite of ideological differences. Forums from under-represented categories, like Entertainment and Technology, correlate less strongly with anything (but more strongly with forums in their own category). It might be worthwile to generate correlation matrices on samples weighted by category. 

The numbers are more meaningful when compared against a specific forum than they are globally. The Atlantic correlates with Breitbart (0.57), but 154 other forums correlate with Breitbart more strongly; likewise, 167 forums are more correlated with The Atlantic. Meanwhile, the top correlation with The AV Club is [Tom and Lorenzo](http://tomandlorenzo.com/) at 0.58.

Here's a broader view of the correlations between the 100 most referenced forums:

<div class="container">
  <div class="row" id="matrix"></div>
</div>
<script type="text/javascript" src="corr_matrix.js"></script>

<p class="caption">Correlation matrix adapted from Karl Broman's [example](https://github.com/kbroman/d3examples/tree/master/corr_w_scatter)</p>

The forums here are sorted by category, with Disqus Channels at the top left and News in the bottom-right quadrant.

### Clustering

Let's see if we can make more sense of the network. 

A correlation matrix looks an awful lot like a fully-connected graph, and graphs lend themselves to all kinds of fun analysis. Let's say each forum is a vertex, and each positive correlation value is an edge. We can use the correlation values to power a force-directed graph, where a high correlation pulls two forums together and a weak or negative one pushes them apart. With a few hundred forums, it looks like this:

<p class="caption">
Click on a circle and drag it around to see how the forums interact.
</p>
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
            <li><a href="#" value='category'>Category</a></li>
            <li><a href="#" value='group'>MCL (e=2, r=3)</a></li>
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
<p class="caption">
Click  on the "Category" and "Clusters With" links to highlight groups in the graph.
</p>
<script src="d3-vis.js"></script>
<script src="interactive-interface.js"></script>

Each circle is a forum, and links are correlations. In this graph, I only included correlations greater than 0.5, and only the top five correlations for each forum. Link strength is based on correlation strength. By default, the color of each circle corresponds to its Disqus category.

Markov Clustering (MCL) is one way to group the forums algorithmically. You can read about it [here](http://micans.org/mcl/), it's fascinating. Basically, you provide a graph with edge weights, and the algorithm manipulates the graph so that each vertex "clusters" around a single other vertex -- possibly itself. There are two parameters, e and r, which control how large the clusters are. You can see the results of the clustering by selecting "Markov clusters" under "Color by..."

What's striking about the graph is how tight some of the regions are. Again, the "News" sites tend to cluster together in the middle, but that's not the whole story. If you scan that big region in the middle, you'll notice a lot of "Culture," "Business," and some "Entertainment" as well. And check out the region around Breitbart (one of the biggest circles, towards the top of the graph).  There are dozens of sites that correlate quite strongly with each other -- the conservative blogosphere. 

Turn on Markov coloring. You can click on the "Clusters with" link under forum details to highlight a cluster in the graph. Breitbart & co. are mostly in blue, though the green group is closely related. Liberal blogs have their own clusters -- [Media Matters](https://mediamatters.org/), [Wonkette](https://wonkette.com/), and [Raw Story](http://www.rawstory.com/) form the core of a tan group towards the left. The orange cluster includes a lot of the "old guard:" The Atlantic, Rolling Stone, and CBS local affiliates, plus, for some reason, a whole bunch of entertainment sites. 

Look at the top correlations for the dark blue forums. Most of them have more than one correlation above .9; Breitbart has five.  The green group is even more tight-knit. Due to the way I added links (a maximum of five per forum), portions of the green group cut themselves off from the rest of the network by being *too* connected. You should see them floating off to the left.  [TotalConservative](http://totalconservative.com/), [UnfilteredPatriot](http://unfilteredpatriot.com/), [PatriotNewsDaily](http://patriotnewsdaily.com/): all these forums have r-values of over .99 with each other. They also correlate with blue-group forums quite well, but don't get a chance to link to them because of link limits. The only other cliques in the graph this tight involve sub-forums of the same site, e.g.  CBS local stations or Disqus channels.

### Activity

Another interesting metric is the number of comments (posts) each forum is actually getting. To estimate, I summed up the number of posts in the top 100 threads for each forum during a 30-day period. By far, the three most active sites are Breitbart, [World Star](http://www.worldstarhiphop.com/videos/), and The Hill -- it's not even close. Each of those forums averages over 7,000 posts per thread on their most popular articles. Due to the way the API is set up, it's tough (several hours per forum) to estimate the *total* number of posts a forum gets in the same period. Out of curiosity, I did pull all the data for Breitbart, and found just over 3 million posts between January 20 and February 20. To put that in perspective, Disqus claims to serve 50 million comments per month. That would mean Breitbart *alone* accounts for over 6% of their traffic.

Something else stuck out to me. Try coloring the graph by "activity," and check out the forums that don't have any recent posts. Any of them sound familiar? CNN, NBC, Bloomberg, Politico... that's not a glitch. Dozens of sites have shut down their Disqus forums in the past couple of years, and a good chunk of them are "mainstream." 

<img src="./media/cnn-empty.png" alt="CNN forum page" class="centered">
<p class="caption">Fifty thousand people used to post here. Now it's a ghost town.</p>

Some of them, like Politico and The National Review, have switched over to Facebook-powered comment machinery. Others, like CNN and Bloomberg, appear to have ditched comments altogether. It's possible this is related to hate speech issues Disqus has had in the past (and has recently [tried to address](https://blog.disqus.com/our-commitment-to-fighting-hate-speech)), or maybe it's part of [the Left's war on comments](http://www.breitbart.com/tech/2015/10/27/the-lefts-war-on-comment-sections/)... I haven't delved deeper, but it's interesting either way.

I hope to have more details about forum activity in the next installment.

### Topics

"Topic modeling" is a class of machine learning algorithms that try to find the abstract "topics" which describe a corpus of text. Suppose you have a set of documents, like a collection of articles from a newspaper. Each article is probably *about* something (or a few different things). Words related to that something will probably show up more often than they do in other documents. An article about World War II will have the words "battle," "fascism" and "Hitler" more often than most other articles will. In contrast, all articles will use lots of generic words like "this," "is," etc. A topic modeler will take a set of documents and pull out groups of words that occur (1) often together in some documents and (2) seldom in others.

For each forum with enough activity, I pulled down raw text from that forum's most active comment threads. I treated each thread like a document and trained a topic model on the set of all threads. I used [Non-negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) to do the modeling, and generated forty topics in all. The rest of the details are beyond the scope of this post, but all the code is on [github](https://github.com/bcyphers/disqus-data).

In the graphic above, the "Top topics" listed for each forum are the most common from all that forum's threads. The score next to each topic is the average "strength" of each topic in all of that forum's documents. You should notice a few topics dominating over a variety of forums: in his first thirty days, people talked a lot about Trump and the government.

The hastily-built chart below shows each of the forty topics, each represented by a rectangle and weighted by how common it is. Hover over each topic bar to see which forums talk about it the most. And please take all of this with a block of salt: I'm very new to NLP and may have made some grave statistical errors along the way. Don't use any of this data as "evidence" to support anything consequential.

<div class="container" id="topics">
<div class="row">
 <div class="col-4" id="topic-graph">
 </div>
 <div class="col" id="topic-list">
   <b id="topics-title">Details</b>
   <ol class="list-group">
   </ol>
 </div>
</div>
</div>
<p class="caption">
Hover over a topic to see which forums talked about it the most.
</p>

With that said, we have some pretty interesting results. 

First, check out the top topics for each site in the network graph above. At [Taki's Mag](http://takimag.com), which flirts with white nationalism, comment threads were mostly about "soros, government, violence" and race. At Occidental Dissent, which is [openly fascist](http://www.occidentaldissent.com/2017/03/23/the-philosophy-of-fascism/), the dominant topic was "israel, jews, jewish, anti, land." [Wired](http://www.wired.com/) readers talk about "science, climate, evidence" more than anything else. And check out [Packers.com](http://packers.com/): it looks like sports chat wasn't common enough to earn its own topic, but the model associated the both the TV-show related "season, shows, new..." and the video game-related "game, games, play..." with the Pack.

Now take a look at the top topics overall. The most common topic is the generic-sounding "really, way, going, make, say," but the second-most common is about Trump and the election. You'll notice topics on trending news stories from that month, like ["flynn, russian, administration, pence, security"](https://www.washingtonpost.com/world/national-security/national-security-adviser-flynn-discussed-sanctions-with-russian-ambassador-despite-denials-officials-say/2017/02/09/f85b29d6-ee11-11e6-b4ff-ac2cf509efe5_story.html) and ["milo, speech, gay, breitbart, right"](http://www.cbsnews.com/news/milo-yiannopoulos-says-he-was-not-supporting-pedophilia/).  The two forums talking most about "black, white, racist..." were the black-focused [Clutch Magazine](http://www.clutchmagonline.com/) and the white-supremicist [American Renaissance](https://www.amren.com/). Talking most about "trump, obama, president..." were political blogs like [The Hill](http://thehill.com/), [The Right Scoop](http://therightscoop.com/), and... [TMZ](http://www.tmz.com/)?  

### Next Steps

There's a lot more that could be done with this data, but for the sake of brevity and time I'm ending here. I'm still actively downloading raw post text and pulling users for new forums. Next time, I'll delve more into the content of the comments sections of each forum. I've got a slew of NLP-related ideas that I'll try to tackle in a coherent way. If you have any questions, corrections or suggestions, please let me know! [bcyphers@mit.edu](mailto:bcyphers@mit.edu)
