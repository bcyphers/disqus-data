function HSVtoRGB(h, s, v) {
    var r, g, b, i, f, p, q, t;
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return {
        r : Math.round(r * 255),
        g : Math.round(g * 255),
        b : Math.round(b * 255),
    };
}

var topicSelected = null;

function buildTopicGraph(data) {
    var hue = 0, sat = 0.5, val = 0.95;
    var goldenRatio = 0.618033988749859;
    var topicMap = {};
    var totalWeight = Object.values(data['_baseline']).reduce(
            function(a, b) { return a + b; }, 0);
    var totalHeight = 500;
    var numForums = 15;

    // build a map from topics to top forums
    for (var t in data['_baseline']) { 
        // sort forums for this topic by score
        var list = Object.keys(data).map(function(f) { return [f, data[f][t]]; });
        list.sort(function(first, second) { return second[1] - first[1]; });
        topicMap[t] = { 
            weight: data['_baseline'][t] / totalWeight,
            forums: list,
        };
    }

    var topList = Object.keys(topicMap).map(
        function(t) { return [t, topicMap[t].weight]; });
    topList.sort(function(first, second) { return second[1] - first[1]; });
    topList = topList.map(function(t) { return t[0]; });

    for (var i = 0; i < numForums + 1; i++) {
        var html = "<li id='forum-" + i + "' class='row'> \
           <div class='col-2 topic-score'></div> \
           <div class='col forum-name'></div> \
         </li>";
        $('#topic-list ol').append(html);
    }

    // bind the data and topics local variables to a function
    var selectTopic = (function(data, topicMap) {
        // update information box with top forums for topic
        return function(topic) {
            var forums = topicMap[topic].forums;
            var baseline = data['_baseline'][topic];

            // set title
            $("#topic-list #topic-name").html('"' + topic + '"');

            // set baseline info
            $("#topic-list #forum-0 .topic-score").html("<i>" + baseline.toFixed(3) + "</i>");
            $("#topic-list #forum-0 .forum-name").html("<i>average</i>");

            for (var j = 0; j < numForums; j++) {
                var forum = forums[j][0];
                if (forums[j][1] <= baseline) {
                  $("#topic-list #forum-" + (j+1) + " .topic-score").html("");
                  $("#topic-list #forum-" + (j+1) + " .forum-name").html("");
                  continue;
                }

                //var value = forums[j][1] / baseline;
                //var num_dec = Math.max(2 - Math.floor(Math.log10(value)), 0);
                //value = value.toFixed(num_dec);

                var value = forums[j][1].toFixed(3);
                var name = details[forum].name;
                if (details[forum].url != null) {
                    name = '<a href="' + details[forum].url + '">' + name + '</a>';
                }

                // update info for this forum
                $("#topic-list #forum-" + (j+1) + " .topic-score").html(value);
                $("#topic-list #forum-" + (j+1) + " .forum-name").html(name);
            }
        }
    })(data, topicMap);
    
    for (var i = 0; i < topList.length; i++) {
        var t = topList[i];
        
        hue = (hue + goldenRatio) % 1;
        var rgb = HSVtoRGB(hue, 0.5, val);
        var cssSoft = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';
        rgb = HSVtoRGB(hue, 0.8, val);
        cssBright = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';

        $('div#topic-graph').append('<div class="topic-box" id="topic-' + i + '"></div>');
        $('div#topic-' + i).css({
            height: Math.round(topicMap[t].weight * totalHeight) + 'px',
            backgroundColor: cssSoft,
        });

        function setColor(color) { 
            return function() { $(this).css('background-color', color); } 
        }

        $('div#topic-' + i).hover(setColor(cssBright), setColor(cssSoft));

        var hoverTopic = (function(topic) { return function(e) { selectTopic(topic); }})(t);

        $('div#topic-' + i).hover(hoverTopic,
          function(e) { if (topicSelected != null) selectTopic(topicSelected); });
        $('div#topic-' + i).click((function(topic) { 
            return function(e) { topicSelected = topic; }
        })(t));
    }
}
