function submitForm(event) {
    event.preventDefault();

    // Initialize an empty array to hold the feature values
    const features =['Yes', '26-100', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Very easy', 'Yes', 'No', 'Yes', 'No', 'No', 'No', "No, I don't know any", 'Yes', 'Yes, they all did', 'N/A (not currently aware)', 'Some did', 'Some did', "I don't know", 'None of them', 'Yes, all of them', 'Some of my previous employers', 'No, at none of my previous employers', 'Yes, they all did', 'None of them', 'No', 'Yes', "No, I don't think it would", 'Yes, I think they would', 'Somewhat open', 'Yes, I experienced', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Not applicable to me', 'Sometimes', '20', 'Male', 'United States of America', 'New York', 'United States of America', 'New York', 'Sometimes', 'Dev Evangelist/Advocate|Back-end Developer'];

    // Loop through all the question elements and parse their values safely
    for (let i = 1; i <= 48; i++) {
        const element = document.querySelector(`select[name="q${i}"], input[name="q${i}"]`);
    
        // Check if the element exists
        if (element) {
            let value = element.value.trim();
            // Check if value is not empty
            if (value !== "") {
                // Ensure that the value is stored as a string
                features.push(value);
            } else {
                console.error(`Value for question q${i} is empty.`);
            }
        } else {
            console.error(`Element for question q${i} not found.`);
        }
    }
    console.log(`features: ${features}`);

    // Send the data to the server
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: features }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const resultElement = document.getElementById('result');
        if (resultElement) {
            if (data.error) {
                console.error('Server Error:', data.error);
                resultElement.innerText = 'An error occurred: ' + data.error;
            } else {
                resultElement.innerText = 'Prediction: ' + data.prediction;
            }
        } else {
            console.error('Result element not found in the HTML.');
        }
    })
    .catch(error => console.error('Error while fetching:', error));
    
}
