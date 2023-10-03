const editableDiv = document.getElementById("editor");
const placeholder = document.getElementById("placeholder");

editableDiv.addEventListener("input", function () {
    if (editableDiv.textContent.length > 0) {
        placeholder.style.display = "none";
    } else {
        placeholder.style.display = "block";
    }
});

function handle(files) {
    if (files.length) {
        var file = files[0];
        var reader = new FileReader();
        reader.onload = function () {
            text = this.result + "\r\n";
            var editor = ace.edit("editor");
            editor.setTheme("ace/theme/dawn");
            editor.setFontSize(16);
            editor.session.setMode("ace/mode/yaml");
            var initialValue = this.result + "\r\n";
            editor.setValue(initialValue);
            var yamlData = jsyaml.load(initialValue);
            editor.getSession().on("change", function () {
                var yamlText = editor.getValue();
                try {
                    var newData = jsyaml.load(yamlText);
                    console.log("Parsed YAML data:", newData);
                } catch (err) {
                    console.error("Error parsing YAML data:", err);
                }
            });
        }
        reader.readAsText(file);
    }
}