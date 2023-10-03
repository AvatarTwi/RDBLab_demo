function controller(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject("未选择文件");
      return;
    }

    const reader = new FileReader();

    reader.onload = function (event) {
      const fileContent = event.target.result;
      try {
        localStorage.setItem("persistedFile", fileContent);
        resolve("文件已成功持久化");
      } catch (error) {
        reject("持久化文件时发生错误：" + error);
      }
    };

    reader.onerror = function (event) {
      reject("文件读取错误：" + event.target.error);
    };

    reader.readAsDataURL(file);
  });
}

const fileInput = document.getElementById("fileInput");
fileInput.addEventListener("change", async (event) => {
  const selectedFile = event.target.files[0];
  try {
    const result = await controller(selectedFile);
    console.log(result); // 输出成功信息
  } catch (error) {
    console.error(error); // 输出错误信息
  }
});
