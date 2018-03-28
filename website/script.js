//fill in people's details here
const people = [
    {
        name:"FirstName Lastname",
        image: "person.jpg" //images will need to be in website/images/people
    },
    {
        name: "First Lastname",
        image: "person.jpg"
    },
    //...
];

function populate() {
    const peopleDiv = document.getElementById("people");
    const peopleHTMLString = people.reduce((acc, obj) => acc + getPersonDiv(obj.name, obj.image), "");
    peopleDiv.innerHTML = peopleHTMLString;
}

function getPersonDiv(name, img) {
    return  `<div class="person flex-center flex-column">
                <img class="person-img" src="./website/images/people/${img}">
                <div class="person-name">${name}</div>
            </div>`
}

window.onload = () => {
    populate();
}