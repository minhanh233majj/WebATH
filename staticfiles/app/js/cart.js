var updateBtns = document.getElementsByClassName('update-cart')


for (i=0;i < updateBtns.length;i++){
    updateBtns[i].addEventListener('click',function(){
        var productId = this.dataset.product
        var action = this.dataset.action
        console.log('productId',productId,'action',action)
        console.log('user: ',user)
        if (user === "AnonymousUser"){
           console.log('user not logged in')
        } else{
            updateUserOrder(productId,action)
        }
    })
}

function updateUserOrder(productId, action) {
    console.log("Gửi dữ liệu:", { productId, action }); // Kiểm tra dữ liệu

    var url = "http://127.0.0.1:8000/update_item/";

    fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({ productId: productId, action: action }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log("Server Response:", data);
        location.reload()
    })
    .catch((error) => {
        console.error("Lỗi:", error);
    });
}

