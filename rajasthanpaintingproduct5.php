<!DOCTYPE html>
<html lang="en">
<head>
<title>Product</title>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="description" content="Wish shop project">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" type="text/css" href="styles/bootstrap4/bootstrap.min.css">
<link href="plugins/font-awesome-4.7.0/css/font-awesome.min.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" type="text/css" href="styles/product.css">
<link rel="stylesheet" type="text/css" href="styles/product_responsive.css">
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
				<li class="menu_mm"><a href="contact.php">Contact</a></li>
			</ul>
		</nav>
	</div>

	<!-- Home -->

	<div class="home">
		<div class="home_background parallax-window" data-parallax="scroll" data-image-src="images/Background/Asia-India-Udaipur-Lake-1200x480.jpg" data-speed="0.8"></div>
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="home_container">
						<div class="home_content">
							<div class="home_title">PRODUCT</div>
							<div class="breadcrumbs">
								<ul>
									<li><a href="http://localhost/hack/complete/index.php">Home</a></li>
									<li>Product</li>
									<li>Product Details</li>
								</ul>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- Product -->
<form action="<?PHP $_SERVER['PHP_SELF']?>" METHOD=POST>
	<div class="product">
		<div class="container">
			
			<div class="row product_row">

				<!-- Product Image -->
				<div class="col-lg-7">
					<div class="product_image">
						<div class="product_image_large"><img src="images/rajasthan/474cc22fa51b0613d16c5c8d17191d23.jpg" alt=""></div>
						
					</div>
				</div>

				<!-- Product Content -->
				<div class="col-lg-5">
					<div class="product_content">
						<div class="product_name">Painting Item</div>
						<div class="product_price">₹1900</div>
						<div class="rating rating_4" data-rating="4">
							<i class="fa fa-star"></i>
							<i class="fa fa-star"></i>
							<i class="fa fa-star"></i>
							<i class="fa fa-star"></i>
							<i class="fa fa-star"></i>
						</div>
						<!-- In Stock -->
						<div class="in_stock_container">
							<div class="in_stock in_stock_true"></div>
							<span>in stock</span>
						</div>
						<div class="product_text">
							<p>Two famous form of textile is bandhani or bandhej like kurtis, suit pieces, sarees, and so on. In jewellery ,meenakari is the most attractive and poplar forms,which are mostly enamelled gold and silver jewellery Most of these carpets, which come in stunning designs, are made and knotted by hands.Different designs of motifs and pastel shades are used to make these durries.There are many craftsmen in Jaipur who prepare blue pottery in Rajasthan by using blue, green and white colors.Wooden Furniture,Jodhpur and Kishangarh are the places where you can get carved casket, screens, doors, chairs, tables, shelf, etc.the famous quilt of rajasthan is Jaipuri rajais which very soft and light.</p>
						</div>
						<!-- Product Quantity -->
						<div class="product_quantity_container">
							<span>Quantity</span>
							<div class="product_quantity clearfix">
								<input id="quantity_input" type="text" pattern="[0-9]*" name=quantity value="1">
								<div class="quantity_buttons">
									<div id="quantity_inc_button" class="quantity_inc quantity_control"><i class="fa fa-caret-up" aria-hidden="true"></i></div>
									<div id="quantity_dec_button" class="quantity_dec quantity_control"><i class="fa fa-caret-down" aria-hidden="true"></i></div>
								</div>
							</div>
						</div>
						<!-- Product Size -->
						<div class="product_size_container">
							<button type="submit" class="review_form_button" NAME="box1" onclick="CLICK">ADD TO CART</button>
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
<script src="js/jquery-3.2.1.min.js"></script>
<script src="styles/bootstrap4/popper.js"></script>
<script src="styles/bootstrap4/bootstrap.min.js"></script>
<script src="plugins/OwlCarousel2-2.2.1/owl.carousel.js"></script>
	<script src="plugins/easing/easing.js"></script>
<script src="plugins/parallax-js-master/parallax.min.js"></script>
<script src="plugins/colorbox/jquery.colorbox-min.js"></script>
<script src="js/custom.js"></script>
</FORM>
<?php
FUNCTION CLICK()
{
$A="Handicraft";
$B=1900;
$C=$_POST["quantity"];
$D=$C*$B;
$E=RAND(1,10000);
$servername="localhost";
$user="root";
$psw="";
$dbname="cart";
$conn=mysqli_connect($servername,$user,$psw,$dbname);
if(!$conn)
{
die("connection failed:".mysqli_connect_error());
echo "TRy again";
}
$sql="insert into shop values('$A',$B,$C,$D,$E)";
$result=mysqli_query($conn,$sql);
$r="select * from shop";
$u=mysqli_query($conn,$r);
$conn->close();
}
IF(ISSET($_POST["box1"]))
{
CLICK();
}

?>
</body>
</html>