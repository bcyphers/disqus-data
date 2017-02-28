$(document).ready(function(){
    var options = $("#options");
    $.each(result, function() {
        options.append($("<option />").val(this.ImageFolderID).text(this.Name));
    });
});
