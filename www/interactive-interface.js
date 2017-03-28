var details = null;
var correlations = null;
var topics = null;
var forumSelected = null;
var subsetSelected = false;

function clearSelection() {
    subsetSelected = false;
    $("circle.background").removeClass("background");
    $("line.background").removeClass("background");
}

categorySelect = function(e) {
    if (subsetSelected) {
        clearSelection();
        return;
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
}

clusterSelect = function(e) {
    if (subsetSelected) {
        clearSelection();
        return;
    }

    subsetSelected = true;

    // when a cluster name is clicked, hilight all nodes in the cluster
    $(".nodes circle").addClass("background");
    $(".links line").addClass("background");
    var group = $("#node-" + forumSelected).attr("group");
    for (var f in details) {
        if ($("#node-" + f).attr("group") == group) {
            $("#node-" + f).removeClass("background");
        }
    }

    var n1, n2;
    $(".links line").each(function(i) {
        n1 = $(this).attr("node1");
        n2 = $(this).attr("node2");
        if ($("#node-" + n1).attr("group") == group && 
            $("#node-" + n2).attr("group") == group) {
            $(this).removeClass("background");
        }
    });
}

$(document).ready(function(){
    $.getJSON("forum-details.json", function(json){ details = json; });
    $.getJSON("forum-correlations.json", function(json){ correlations = json; });
    $.getJSON("forum-topics.json", function(json){ topics = json; });

    function updateDescription(forum) {
        if (details != null) {
            var deets = details[forum];
            var group_id = $("#node-" + forum).attr("group");
            var short_url = "";
            if (deets.url != null) {
                var short_url = deets.url.length > 40 ? 
                    deets.url.substring(0, 38) + "..." : deets.url;
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
                details[group_id].name + "</a>");

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

        if (correlations != null) {
            var cors = correlations[forum];

            // sort other forums by correlation
            // javascript why are you like you are
            var list = Object.keys(cors).map(function(key) { return [key, cors[key]]; });
            list.sort(function(first, second) { return second[1] - first[1]; });

            var i = 0, j = 0;
            while (j < 5) {
                var name = list[i][0];
                if (name == forum) {
                    i += 1;
                    continue;
                }

                if (details != null) { name = details[name].name; }
                if (name.length >= 16) { name = name.substring(0, 15) + "..."; }

                var cor_value = (Math.round(list[i][1] * 100) / 100).toFixed(2);

                if (cor_value > 0) {
                    $("#correlation-" + j).html(cor_value + " " + name);
                } else {
                    $("#correlation-" + j).html("");
                }

                i++;
                j++;
            }
        }

        if (topics != null) {
            if (forum in topics) {
                var tops = topics[forum];

                // sort other forums by correlation
                // javascript why are you like you are
                var list = Object.keys(tops).map(function(key) { return [key, tops[key]]; });
                list.sort(function(first, second) { return second[1] - first[1]; });

                $("#topics-title").html("Top topics");
                var i = 0;
                while (i < 5) {
                    var name = list[i][0];
                    var num_dec = 2 - Math.floor(Math.log10(list[i][1]));
                    var value = list[i][1].toFixed(num_dec);

                    if (value > 1) {
                        $("#topic-" + i + " .topic-score").html(value);
                        $("#topic-" + i + " .topic-name").html(name);
                    } else {
                        $("#topic-" + i + " .topic-score").html("");
                        $("#topic-" + i + " .topic-name").html("");
                    }

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

    $("circle").hover(function(e) {
        forum = e.target.id.replace("node-", "");
        updateDescription(forum);
    }, function(e) {
        updateDescription(forumSelected);
    });

    $("circle").click(function(e) {
        forumSelected = e.target.id.replace("node-", "");
        updateDescription(forumSelected);
        clearSelection();
        $("circle.selected").removeClass("selected");
        $("#" + e.target.id).addClass("selected");
    });

    $("ul#coloring-select li a").click(function(e) {
        // when a selection is made from the "coloring" dropdown, recolor all 
        // the nodes
        var nodes = d3.select("svg").selectAll("g circle");
        var color = d3.scaleOrdinal(d3.schemeCategory20);
        key = e.target.getAttribute("value");

        nodes.each(function(d, i) {
            n = d3.select(this);
            
            if (key == "group")
                n.attr("fill", color(d.group));
            if (key == "category")
                n.attr("fill", color(details[d.id].category));
            if (key == "activity") {
                if (details[d.id].activity > 0)
                    n.attr("fill", "green");
                else
                    n.attr("fill", "red");
            }
        });
    });
});
