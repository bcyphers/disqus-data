var details = null;
var correlations = null;
var topics = null;
var forumSelected = null;
var subsetSelected = false;

function updateDescription(forum) {
    if (details != null) {
        var deets = details[forum];
        var short_url = "";
        if (deets.url != null) {
            var short_url = deets.url.length > 33 ? 
                deets.url.substring(0, 30) + "..." : deets.url;
        }

        var act_str, alexa_str;
        if (deets.activity > 0)
            act_str = Number(deets.activity).toLocaleString() + " posts";
        else
            act_str = "None";

        if (deets.alexa > 0)
            alexa_str = Number(deets.alexa).toLocaleString(); 
        else
            alexa_str = "Unavailable";

        $("#forum-name").html(deets.name);
        $("#detail-url").html('<a href="' + deets.url + '">' + short_url + '</a>');
        $("#detail-category").html('Category: <a id="category-select" href="#">' + 
                deets.category + "</a>");
        $("#detail-cluster").html('Clusters with: <a id="cluster-select" href="#">' +
            details[deets.group].name + "</a>");

        $("#category-select").click(categorySelect);
        $("#cluster-select").click(clusterSelect);
        
        $("#detail-activity").html("Activity (30d): <b>" + act_str + "</b>");
        $("#detail-connectivity").html("Alexa rank: <b>" + alexa_str + "</b>");

        if (deets.description != null && 
            deets.description != "None" && 
            deets.description.length > 0) {
            $("#detail-description").html("<i>" + deets.description.trim() + "</i>");
        } else {
            $("#detail-description").html("No description available.");
        }
    } else {
        $("#forum-name").html(forum);
    }

    // update top 5 correlations for the selected forum
    if (correlations != null) {
        var forum_ix = correlations.index.indexOf(forum);
        var cors = correlations.data[forum_ix];

        // sort other forums by correlation
        var list = [];
        for (var i = 0; i < cors.length; i++) {
            list.push([correlations.index[i], cors[i]]);
        }
        list.sort(function(first, second) { return second[1] - first[1]; });

        // list top 5 correlations for this forum
        var i = 0, j = 0;
        while (j < 5) {
            var cor_frm = list[i][0];
            if (cor_frm == forum) {
                i += 1;
                continue;
            }

            if (details != null) { name = details[cor_frm].name; }
            if (name.length >= 16) { name = name.substring(0, 15) + "..."; }

            var cor_value = (Math.round(list[i][1] * 100) / 100).toFixed(2);

            if (cor_value > 0) {
                $("#correlation-" + j).html(cor_value + " " + 
                        '<a href="#" forum="' + cor_frm + '">' + name + "</a>");
            } else {
                $("#correlation-" + j).html("");
            }

            i++;
            j++;
        }
        
        // onclick handler for the correlation list
        $("#correlate-ol li a").click(function(e) {
            e.preventDefault();
            var forum = $(e.target).attr("forum");  
            console.log(forum);
            forumSelect(forum);
            return false;
        });
    }

    // list top 3 topics for this forum
    if (topics != null) {
        if (forum in topics) {
            var tops = topics[forum];

            // sort topics by score
            var list = Object.keys(tops).map(function(key) { 
                return [key, tops[key]]; 
            });
            list.sort(function(first, second) { return second[1] - first[1]; });

            $("#topics-title").html("Top topics");
            var i = 0;
            while (i < 3) {
                var name = list[i][0];
                var value = list[i][1].toFixed(3).slice(1);

                $("#topic-" + i + " .topic-score").html(value);
                $("#topic-" + i + " .topic-name").html(name);

                i++;
            }
        } else {
            $("#topics-title").html("No topic data available");
            for (var i = 0; i < 5; i++) {
                $("#topic-" + i + " .topic-score").html("");
                $("#topic-" + i + " .topic-name").html("");
            }
        }
    }
}

function clearSelection() {
    subsetSelected = false;
    $("circle.background").removeClass("background");
    $("line.background").removeClass("background");
}

function forumSelect(forum) {
    forumSelected = forum;
    updateDescription(forumSelected);
    clearSelection();
    $("circle.selected").removeClass("selected");
    $("#node-" + forum).addClass("selected");
}

function categorySelect(e) {
    e.preventDefault();

    if (subsetSelected) {
        clearSelection();
        return false;
    }

    subsetSelected = true;

    // when a category name is clicked, hilight all nodes in the category
    $(".nodes circle").addClass("background");
    $(".links line").addClass("background");

    var cat = details[forumSelected].category;
    for (var f in details) {
        if (details[f].category == cat) {
            $("#node-" + f).removeClass("background");
        }
    }

    var n1, n2;
    $(".links line").each(function(i) {
        n1 = $(this).attr("node1");
        n2 = $(this).attr("node2");
        if (details[n1].category == cat && details[n2].category == cat) {
            $(this).removeClass("background");
        }
    });
    return false;
}

function recolorCircles(key) {
    // when a selection is made from the "coloring" dropdown, recolor all 
    // the nodes
    var nodes = d3.select("svg#force-directed").selectAll("g circle");
    var color = d3.scaleOrdinal(d3.schemeCategory20);

    nodes.each(function(d, i) {
        n = d3.select(this);
        
        if (key == "group")
            n.attr("fill", color(details[d.id].group));
        if (key == "category")
            n.attr("fill", color(details[d.id].category));
        if (key == "activity") {
            if (details[d.id].activity > 0)
                n.attr("fill", "green");
            else
                n.attr("fill", "red");
        }
    });
}

function clusterSelect(e) {
    e.preventDefault();

    if (subsetSelected) {
        clearSelection();
        return false;
    }

    subsetSelected = true;

    // when a cluster name is clicked, hilight all nodes in the cluster
    $(".nodes circle").addClass("background");
    $(".links line").addClass("background");
    var group = details[forumSelected].group;
    for (var f in details) {
        if (details[f].group == group) {
            $("#node-" + f).removeClass("background");
        }
    }

    var n1, n2;
    $(".links line").each(function(i) {
        n1 = $(this).attr("node1");
        n2 = $(this).attr("node2");
        if (details[n1].group == group && 
            details[n2].group == group) {
            $(this).removeClass("background");
        }
    });
    return false;
}

$(document).ready(function(){
    var d3Ready = false,
        detailsReady = false;

    $.getJSON("data/correlations.json", function(json){ correlations = json; });
    $.getJSON("data/topics.json", function(json) { 
        topics = json; 
        buildTopicGraph(topics);
    });
    $.getJSON("data/details.json", function(json) { 
        details = json; 
        detailsReady = true;
        finalSetup();
    });
    $.getJSON("data/force-graph-init.json", function(json) {
      forceGraphSimulate("data/force-graph-4-05.json", d3Callback, json);
    });

    //forceGraphSimulate("data/05-22/force-graph.json", d3Callback, null);
    function finalSetup() {
        if (detailsReady && d3Ready) {
            recolorCircles("category");
            forumSelect("theatlantic");
        }
    }

    function d3Callback() {
        $(".nodes circle").hover(function(e) {
            var forum = e.target.id.replace("node-", "");
            updateDescription(forum);
        }, function(e) {
            updateDescription(forumSelected);
        });

        $(".nodes circle").click(function(e) {
            e.preventDefault();
            var forum = e.target.id.replace("node-", "")   
            forumSelect(forum);
            return false;
        });

        $("#coloring-select li a").click(function(e) {
            e.preventDefault();
            recolorCircles(e.target.getAttribute("value"));
            return false;
        });

        d3Ready = true;
        finalSetup();
    }
});

function saveGraphPositions() {
    var graph = {};
    $('.nodes circle').each(function() {
        graph[$(this).attr('id')] = {
            x: $(this).attr('cx'), 
            y: $(this).attr('cy')
        };
    });

    localStorage.setItem('graph.json', JSON.stringify(graph));
}
