/* Upload page interactivity. */

const fileInput  = document.getElementById("file-input");
const dropZone   = document.getElementById("drop-zone");
const preview    = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const clearBtn   = document.getElementById("clear-btn");
const submitBtn  = document.getElementById("submit-btn");
const form       = document.getElementById("upload-form");

function showPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  preview.hidden = false;
  clearBtn.hidden = false;
  submitBtn.disabled = false;
}

function clearPreview() {
  previewImg.src = "";
  preview.hidden = true;
  clearBtn.hidden = true;
  submitBtn.disabled = true;
  fileInput.value = "";
}

fileInput.addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) showPreview(f);
});

clearBtn.addEventListener("click", (e) => {
  e.preventDefault();
  clearPreview();
});

["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) {
    fileInput.files = e.dataTransfer.files;
    showPreview(f);
  }
});

form.addEventListener("submit", () => {
  submitBtn.classList.add("loading");
  submitBtn.disabled = true;
});