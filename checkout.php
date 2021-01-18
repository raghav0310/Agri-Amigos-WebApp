<!DOCTYPE html>
<html lang="en">
<head>
<title>Checkout</title>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="description" content="Wish shop project">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" type="text/css" href="styles/bootstrap4/bootstrap.min.css">
<link href="plugins/font-awesome-4.7.0/css/font-awesome.min.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" type="text/css" href="styles/checkout.css">
<link rel="stylesheet" type="text/css" href="styles/checkout_responsive.css">
</head>
<body>

<div class="super_container">
	
	<!-- Header -->

	<header class="header">
		<div class="header_inner d-flex flex-row align-items-center justify-content-start">
			<div class="logo">
				<p><img src="images/Background/ART.png" height=80 width=250></p>
            </div>
			<nav class="main_nav">
				<ul>
					<li><a href="http://localhost/hack/complete/index.php">home</a></li>
					<li><a href="http://localhost/hack/complete/index.php#handi">Handicrafts</a></li>
					<li><a href="http://localhost/hack/complete/index.php#jewel">Jewellery</a></li>
					<li><a href="http://localhost/hack/complete/index.php#paint">Paintings</a></li>
					<li><a href="contact.php">Contact</a></li>
				</ul>
			</nav>
			<div class="header_content ml-auto">
				<div class="search header_search">
					<form action="#">
						<input type="search" class="search_input" required="required">
						<button type="submit" id="search_button" class="search_button"><img src="images/magnifying-glass.svg" alt=""></button>
					</form>
				</div>
				<div class="shopping">
					<!-- Cart -->
					<a href="#">
						<div class="cart">
							<img src="images/shopping-bag.svg" alt="">
							<div class="cart_num_container">
								<div class="cart_num_inner">
									<div class="cart_num">1</div>
								</div>
							</div>
						</div>
					</a>
					<!-- Star -->
					<a href="#">
						<div class="star">
							<img src="images/star.svg" alt="">
							<div class="star_num_container">
								<div class="star_num_inner">
									<div class="star_num">0</div>
								</div>
							</div>
						</div>
					</a>
					<!-- Avatar -->
					<a href="cart.php">
						<div class="avatar">
							<img src="images/avatar.svg" alt="">
						</div>
					</a>
				</div>
			</div>

			<div class="burger_container d-flex flex-column align-items-center justify-content-around menu_mm"><div></div><div></div><div></div></div>
		</div>
	</header>

	<!-- Menu -->

	<div class="menu d-flex flex-column align-items-end justify-content-start text-right menu_mm trans_400">
		<div class="menu_close_container"><div class="menu_close"><div></div><div></div></div></div>
		<div class="logo menu_mm"><img src="images/Background/ART.png" height=80 width=250></div>
		<div class="search">
			<form action="#">
				<input type="search" class="search_input menu_mm" required="required">
				<button type="submit" id="search_button_menu" class="search_button menu_mm"><img class="menu_mm" src="images/magnifying-glass.svg" alt=""></button>
			</form>
		</div>
		<nav class="menu_nav">
			<ul class="menu_mm">
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php">Home</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php#handi">Handicraft</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php#jewel">Jewellery</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php#paint">Painting</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.phpcontact.php">Contact</a></li>
			</ul>
		</nav>
	</div>

	<!-- Home -->

	<div class="home">
		<div class="home_background parallax-window" data-parallax="scroll" data-image-src="images/Background/Kutch-Holiday-Guide.jpg" data-speed="0.8"></div>
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="home_container">
						<div class="home_content">
							<div class="home_title">Checkout</div>
							<div class="breadcrumbs">
								<ul>
									<li><a href="index.html">Home</a></li>
									<li><a href="index.html">Shopping Cart</a></li>
									<li>Shopping Cart</li>
								</ul>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- Checkout -->

	<div class="checkout">
		<div class="container">
			<div class="row">

				<!-- Billing Details -->
				<div class="col-lg-6">
					<div class="billing">
						<div class="checkout_title">billing details</div>
						<div class="checkout_form_container">
							<form method=POST action="<?PHP $_SERVER['PHP_SELF']?>" id="checkout_form">
								<div class="d-flex flex-lg-row flex-column align-items-start justify-content-between">
									<input type="text" name="fname" class="checkout_input checkout_input_50" placeholder="First Name" required="required">
									<input type="text" class="checkout_input checkout_input_50" placeholder="Last Name" name="lname" required="required">
								</div>
								<input type="text" class="checkout_input" name="mail" placeholder="E-mail" required="required">
								<select name="country" id="country" class="country_select checkout_input">
									<option>India</option>
									<option>Nepal</option>
									<option>Bangladesh</option>
									<option>Europe</option>
									<option>Honkong</option>
								</select>
								<input type="text" name="address" class="checkout_input" placeholder="Address" required="required">
								<input type="text" class="checkout_input" name="town" placeholder="Town" required="required">
								<div class="d-flex flex-lg-row flex-column align-items-start justify-content-between">
									<input type="text" class="checkout_input checkout_input_50" name="code" placeholder="Zipcode" required="required">
									<input type="text" class="checkout_input checkout_input_50" placeholder="Phone No" required="required" name="mobile">
								</div>
								<textarea name="checkout_comment" id="checkout_comment" class="checkout_comment" placeholder="Leave a comment about your order"></textarea>
								
							</form>
						</div>
					</div>
				</div>

				<!-- Cart Details -->
				<div class="col-lg-6">
					<div class="cart_details">
						<div class="checkout_title">cart total</div>
						<div class="cart_total">
							<ul>
								<li class="d-flex flex-row align-items-center justify-content-start">
									<div class="cart_total_title">Product</div>
									<div class="cart_total_price ml-auto">Total</div>
								</li>
								
								<li class="d-flex flex-row align-items-center justify-content-start">
									<div class="cart_total_title">Subtotal</div>
									<div class="cart_total_price ml-auto">Rs. 7000</div>
								</li>
								<li class="d-flex flex-row align-items-center justify-content-start">
									<div class="cart_total_title">Shipping</div>
									<div class="cart_total_price ml-auto">Rs. 50.00</div>
								</li>
								<li class="d-flex flex-row align-items-start justify-content-start total_row">
									<div class="cart_total_title">Total</div>
									<div class="cart_total_price ml-auto">Rs. 7050</div>
								</li>
							</ul>
						</div>
						<div class="payment_options">
							<!--<div>
								<input type="radio" id="radio_payment_1" name="regular_radio" class="regular_radio">
								<label for="radio_payment_1">cash on delivery</label>
							</div>-->
							<div>
								<input type="radio" id="radio_payment_2" name="regular_radio" class="regular_radio" checked>
								<label for="radio_payment_2">paytm</label>
								<!--<div class="visa payment_option"><a href="#"><img src="images/visa.jpg" alt=""></a></div>
								<div class="master payment_option"><a href="#"><img src="images/master.jpg" alt=""></a></div>-->
							</div>
							<button name="placeorder" class="cart_total_button"><a href=https://paytm.com/paytmwallet>place order</a></button>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>

	
<!-- Footer -->

	<footer class="footer">
		<div class="container">
			<div class="row">
				<div class="col text-center">
					<div class="footer_logo"><a href="#">Artistery</a></div>
					<nav class="footer_nav">
						<ul>
							<li><a href="index.html">Home</a></li>
							<li><a href="categories.html">Handicraft</a></li>
							<li><a href="categories.html">Jewellery</a></li>
							<li><a href="categories.html">Paintings</a></li>
							<li><a href="contact.html">contact</a></li>
						</ul>
					</nav>
					<div class="footer_social">
						<ul>
							<li><a href="#"><i class="fa fa-pinterest" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-linkedin" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-instagram" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-reddit-alien" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-twitter" aria-hidden="true"></i></a></li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	</footer>
</div>
<hr color="#DB7115"><hr color="#DB7115">
</div>

<script src="js/jquery-3.2.1.min.js"></script>
<script src="styles/bootstrap4/popper.js"></script>
<script src="styles/bootstrap4/bootstrap.min.js"></script>
<script src="plugins/easing/easing.js"></script>
<script src="plugins/parallax-js-master/parallax.min.js"></script>
<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&key=AIzaSyCIwF204lFZg1y4kPSIhKaHEXMLYxxuMhA"></script>
<script src="js/contact_custom.js"></script>
<?php
FUNCTION CLICK()
{
$A=$_POST["fname"];
$B=$_POST["lname"];
$C=$_POST["mail"];
$D=$_POST["country"];
$E=$_POST["address"];
$F=$_POST["town"];
$G=$_POST["code"];
$H=$_POST["mobile"];
$I=$_POST["checkout_comment"];
$servername="localhost";
$user="root";
$psw="";
$dbname="contact";
$conn=mysqli_connect($servername,$user,$psw,$dbname);
if(!$conn)
{
die("connection failed:".mysqli_connect_error());
echo "TRy again";
}
$sql="insert into info values('$A','$B','$C','$D','$E','$F',$G,$H,'$I')";
$result=mysqli_query($conn,$sql);
$r="select * from info";
$u=mysqli_query($conn,$r);
$conn->close();
}
IF(ISSET($_POST["placeorder"]))
{
CLICK();
}
?>
</body>
</html>