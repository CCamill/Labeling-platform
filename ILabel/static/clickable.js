// //Determines if the mouse was pressed on the previous frame
// var cl_mouseWasPressed = false;
// //Last hovered button
// var cl_lastHovered = null;
// //Last pressed button
// var cl_lastClicked = null;
//All created buttons
var cl_clickables = [];

//This function is what makes the magic happen and should be ran after
//each draw cycle.
p5.prototype.runGUI = function () {
    for (i = 0; i < cl_clickables.length; ++i) {
        // //Determines if the mouse was pressed on the previous frame
        let cl_mouseWasPressed = cl_clickables[i].cl_mouseWasPressed;
        // //Last hovered button
        let cl_lastHovered = cl_clickables[i].cl_lastHovered;
        // //Last pressed button
        let cl_lastClicked = cl_clickables[i].cl_lastClicked;

        if (cl_lastHovered != cl_clickables[i])
            cl_clickables[i].onOutside();

        if (cl_lastHovered != null) {
            if (cl_lastClicked != cl_lastHovered) {
                cl_lastHovered.onHover();
            }
        }
        if (cl_mouseWasPressed && cl_lastClicked != null) {
            cl_lastClicked.onPress();
        }
        if (cl_mouseWasPressed && !mouseIsPressed && cl_lastClicked != null) {
            if (cl_lastClicked == cl_lastHovered) {
                cl_clickables[i].cl_lastClicked.onRelease();
            }
            cl_clickables[i].cl_lastClicked = null;
        }
        cl_clickables[i].cl_lastHovered = null;
        cl_clickables[i].cl_mouseWasPressed = mouseIsPressed;
    }
}

p5.prototype.registerMethod('post', p5.prototype.runGUI);

//This function is used to get the bounding size of a
//string of text for use in the 'textScaled' property
function getTextBounds(m, font, size) {
    let txt = document.createElement("span");
    document.body.appendChild(txt);

    txt.style.font = font;
    txt.style.fontSize = size + "px";
    txt.style.height = 'auto';
    txt.style.width = 'auto';
    txt.style.position = 'absolute';
    txt.style.whiteSpace = 'no-wrap';
    txt.innerHTML = m;

    let width = Math.ceil(txt.clientWidth);
    let height = Math.ceil(txt.clientHeight);
    document.body.removeChild(txt);
    return [width, height];
}

//Button Class
function Clickable() {
    this.x = 0; //X position of the clickable
    this.y = 0; //Y position of the clickable
    this.im_w = 0;
    this.im_h = 0;
    this.im_scale = 1;
    this.scaled_w = 0
    this.scaled_h = 0
    this.innerX = 0;
    this.innerY = 0;
    this.isPressing = false;

    this.width = 100; //Width of the clickable
    this.height = 50; //Height of the clickable
    this.color = "#FFFFFF"; //Background color of the clickable
    this.cornerRadius = 10; //Corner radius of the clickable
    this.strokeWeight = 1; //Stroke width of the clickable
    this.stroke = "#000000"; //Border color of the clickable
    this.text = ""; //Text of the clickable
    this.textColor = "#000000"; //Color for the text shown
    this.textSize = 12; //Size for the text shown
    this.textFont = "sans-serif"; //Font for the text shown
    this.textScaled = false; //Scale the text with the size of the clickable
    this.visiable = true;
    // image options
    this.image = null; // image object from p5loadimage()
    this.tint = null; // tint image using color
    this.noTint = true; // default to disable tinting
    this.filter = null; // filter effect

    this.cl_mouseWasPressed = false;
    this.cl_lastHovered = null;
    this.cl_lastClicked = null;

    this.updateTextSize = function () {
        if (this.textScaled) {
            for (let i = this.height; i > 0; i--) {
                if (getTextBounds(this.text, this.textFont, i)[0] <= this.width && getTextBounds(this.text, this.textFont, i)[1] <= this.height) {
                    console.log("textbounds: " + getTextBounds(this.text, this.font, i));
                    console.log("boxsize: " + this.width + ", " + this.height);
                    this.textSize = i / 2;
                    break;
                }
            }
        }
    }
    this.updateTextSize();

    this.onHover = function () {
        //This function is ran when the clickable is hovered but not
        //pressed.
    }

    this.onOutside = function () {
        //This function is ran when the clickable is NOT hovered.
    }

    this.onPress = function () {
        //This fucking is ran when the clickable is pressed.
    }

    this.onRelease = function () {
        //This funcion is ran when the cursor was pressed and then
        //released inside the clickable. If it was pressed inside and
        //then released outside this won't work.
    }

    this.locate = function (x, y) {
        this.x = x;
        this.y = y;
    }

    this.resize = function (w, h) {
        this.width = w;
        this.height = h;
        this.updateTextSize();
    }

    this.drawImage = function () {
        // this.im_h = this.image.height
        // this.im_w = this.image.width
        let _im_scale = Math.floor(Math.min(this.im_scale * this.height / this.im_h, this.im_scale * this.width / this.im_w))

        let scaled_w = _im_scale * this.im_w
        let scaled_h = _im_scale * this.im_h
        this.scaled_w = scaled_w
        this.scaled_h = scaled_h
        image(this.image, (this.width - scaled_w) / 2, (this.height - scaled_h) / 2, scaled_w, scaled_h);
        if (this.tint && !this.noTint) {
            tint(this.tint)
        } else {
            noTint();
        }
        if (this.filter) {
            filter(this.filter);
        }
    }

    this.setImage = img => {
        this.image = img
        this.im_h = img.height
        this.im_w = img.width
        let _im_scale = Math.floor(Math.min(this.im_scale * this.height / this.im_h, this.im_scale * this.width / this.im_w))

        this.scaled_w = _im_scale * this.im_w
        this.scaled_h= _im_scale * this.im_h
    }

    this.draw = function () {
        this.innerX = mouseX - this.x;
        this.innerY = mouseY - this.y;

        if (this.visiable) {
            fill(this.color);
            stroke(this.stroke);
            strokeWeight(this.strokeWeight);
            rect(this.x - this.width / 2, this.y - this.height / 2, this.width, this.height, this.cornerRadius);
            fill(this.textColor);
            noStroke();

            if (this.image) {
                this.drawImage();
            }
            textAlign(CENTER, CENTER);
            textSize(this.textSize);
            textFont(this.textFont);
            text(this.text, this.x + this.width / 2, this.y + this.height / 2);
        }
        if (Math.abs(mouseX - this.x) <= this.width && Math.abs(mouseY - this.y) <= this.height) {
            this.cl_lastHovered = this;
            if (mouseIsPressed && !this.cl_mouseWasPressed)
                this.cl_lastClicked = this;
        }
    }

    cl_clickables.push(this);
}
