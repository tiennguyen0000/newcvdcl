document.getElementById('upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const uploadedImage = document.getElementById('uploaded-image');
        uploadedImage.src = e.target.result;
        uploadedImage.style.display = 'block';
    };

    if (file) {
        reader.readAsDataURL(file);
    }
});

async function query_gen(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev",
		{
			headers: {
				"Authorization": "Bearer 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",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.blob();
	return result;
}

document.getElementById('generate').addEventListener('click', function() {
    const textInput = document.getElementById('text-input').value;

    console.log("Sinh ảnh từ văn bản:", textInput);

    query_gen(textInput)
    .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
    
        // Display the image
        const img = document.getElementById('generated-image');
        img.src = imageUrl;
        img.style.display = 'block';

    })
});




// increase and decrease
async function query_inde(base64Image, promt, promt_fw, scale) {
    
    // Gửi request và nhận kết quả trả về
    const response = await fetch(
        `https://c4cf-34-106-255-115.ngrok-free.app/i2i/${scale}`,
        {   
            method: "POST",
            headers: {
                'ngrok-skip-browser-warning': 'true',
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                prompt: promt,
                prompt_fw: promt_fw,
                img_base64: base64Image,
            }),
        }
    );

    // Xử lý kết quả trả về dưới dạng JSON
    const result = await response.json();
    return result;
}

document.getElementById('increase').addEventListener('click', function() {
    const uploadedImage = document.getElementById('upload');

    const file = uploadedImage.files[0];
    if (!file) {
        alert("Please select an image first!");
        return;
    }

    // Chuyển ảnh sang base64
    const base64Image = toBase64(file);
    const promt = document.getElementById('promt-input').value;
    const promt_fw = document.getElementById('promtfw-input').value;

    // Gọi API
    scale = "increase";
    base64Image.then(base64 => {
        query_inde(base64, promt, promt_fw, scale)
        .then(data => {
            
            img_base64 = data;
           
            // Hiển thị ảnh kết quả
            const imgElement = document.getElementById('inde-image');
            imgElement.src = `data:image/jpeg;base64,${img_base64}`;
            imgElement.style.display = 'block';
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });

    function toBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(",")[1]); // Lấy base64 sau dấu phẩy
            reader.onerror = (error) => reject(error);
            reader.readAsDataURL(file);
        });
    }
});

document.getElementById('decrease').addEventListener('click', function() {
    const uploadedImage = document.getElementById('upload');

    const file = uploadedImage.files[0];
    if (!file) {
        alert("Please select an image first!");
        return;
    }

    // Chuyển ảnh sang base64
    const base64Image = toBase64(file);
    const promt = document.getElementById('promt-input').value;
    const promt_fw = document.getElementById('promtfw-input').value;

    // console.log(promt, promt_fw, base64Image.json);
    // Gọi API
    scale = "decrease";
    base64Image.then(base64 => {
        query_inde(base64, promt, promt_fw, scale)
        .then(data => {
       
            img_base64 = data;


            // Hiển thị ảnh kết quả
            const imgElement = document.getElementById('inde-image');
            imgElement.src = `data:image/jpeg;base64,${img_base64}`;
            imgElement.style.display = 'block';
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });

    function toBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(",")[1]); // Lấy base64 sau dấu phẩy
            reader.onerror = (error) => reject(error);
            reader.readAsDataURL(file);
        });
    }
});




// reset
document.getElementById('reset_gen').addEventListener('click', function() {
    const Image = document.getElementById('generated-image');
    Image.style.display = 'none';
    Image.src = '';
});
document.getElementById('reset_inde').addEventListener('click', function() {
    const Image1 = document.getElementById('inde-image');
    Image.style.display = 'none';
    Image.src = '';
});







