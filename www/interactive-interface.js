var details = null;
var topics = null;
var correlations = null;
var forumSelected = null;

$(document).ready(function(){
    $.getJSON("forum-details.json", function(json){ details = json; });
    $.getJSON("forum-topics.json", function(json){ topics = json; });
    //$.getJSON("forum-correlations.json", function(json){ correlations = json; });

    function updateDescription(forum) {
        if (details != null) {
            deets = details[forum];

            $("#forum-name").html(deets.name);
            $("#detail-url").html('<a href="' + deets.url + '">' + deets.url + '</a>');
            $("#detail-category").html("Category: <b>" + deets.category + "</b>");
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
