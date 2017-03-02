var details = null;
var correlations = null;
var topics = null;
var forumSelected = null;

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

            $("#forum-name").html(deets.name);
            $("#detail-url").html('<a href="' + deets.url + '">' + short_url + '</a>');
            $("#detail-category").html('Category: <a href="#">' + deets.category + "</a>");
            $("#detail-group").html('Grouped with: <a href="#">' +
                details[group_id].name + "</a>");
            $("#detail-activity").html("Activity (30d): <b>" +
                deets.activity + " posts</b>");
            $("#detail-connectivity").html("Connectivity: <b>" + 
                Math.round(10 * deets.connectivity) / 10 + "</b>");

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
        $(".circle-selected").removeClass("circle-selected");
        $("#" + e.target.id).addClass("circle-selected");
    });
});
