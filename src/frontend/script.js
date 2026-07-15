async function uploadPDF(){

    let file = document.getElementById(
        "pdfFile"
    ).files[0];


    let formData = new FormData();

    formData.append(
        "file",
        file
    );


    let response = await fetch(
        "http://localhost:8000/upload",
        {
            method:"POST",
            body:formData
        }
    );


    let result = await response.json();

    console.log(result);
}

async function ask(){
    let question=document.getElementById(
        "question").value;

    let response = await fetch(
        "http://localhost:8000/chat",
        {method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                query:question
            })
        
        }
    )
    let result = await response.json();
    document.getElementById(
        "answer"
    ).innerText = result.answer;
}